/*
  Copyright© 2024 Raúl Wolters(1)

  This file is part of libinflx_rs (rust bindings for inflatox).

  inflatox is free software: you can redistribute it and/or modify it under
  the terms of the European Union Public License version 1.2 or later, as
  published by the European Commission.

  inflatox is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE. See the European Union Public License for more details.

  You should have received a copy of the EUPL in an/all official language(s) of
  the European Union along with inflatox. If not, see
  <https://ec.europa.eu/info/european-union-public-licence_en/>.

  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/

use ndarray as nd;
use numpy::{PyReadonlyArray1, PyReadwriteArray2};
use pyo3::{pyfunction, PyRef, PyResult};

use crate::{
  dylib::{ExFn3, InflatoxDylib},
  BADGE_PANIC,
};

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

#[derive(Clone)]
struct EoM<'a> {
  lib: &'a InflatoxDylib,
  pars: &'a [f64],
  eqs: Box<[ExFn3]>,
}

impl<'a> EoM<'a> {
  #[inline]
  pub fn new(lib: &'a InflatoxDylib, pars: &'a [f64]) -> Result<Self> {
    assert_eq!(pars.len(), lib.n_pars());
    Ok(EoM { lib, pars, eqs: lib.get_eom()? })
  }

  #[inline]
  /// Safety: this function assumes that `x` and `xdot` are at least `n_fields` long
  pub unsafe fn f(&self, field_idx: usize, x: &[f64], xdot: &[f64], hubble: f64) -> f64 {
    -(self.eqs[field_idx])(x.as_ptr(), xdot.as_ptr(), self.pars.as_ptr())
      - 3. * hubble * xdot[field_idx]
  }

  #[inline]
  /// Safety: this function assumes that `x` is at least `n_fields` long
  pub unsafe fn g(&self, x: &[f64], hubble: f64) -> f64 {
    self.lib.potential()(x.as_ptr(), self.pars.as_ptr()) - 3. * hubble.powi(2)
  }

  #[cfg(test)]
  pub fn test_instance(lib: &'a InflatoxDylib) -> Self {
    unsafe extern "C" fn test_eom(x: *const f64, xdot: *const f64, _pars: *const f64) -> f64 {
      23.0
    }
    let eqs = vec![test_eom as ExFn3].into_boxed_slice();
    let pars = Box::<[_; 0]>::leak(Box::new([]));
    EoM { lib, pars, eqs }
  }
}

struct RK4Solver<'a, const ORDER: usize> {
  eom: EoM<'a>,
  dt: f64,
  a: &'static [&'static [f64]; ORDER],
  k: Vec<[f64; ORDER]>,
  phi_dot: Vec<[f64; ORDER]>,
  l: [f64; ORDER],
  scratch1: Vec<f64>,
  scratch2: Vec<f64>,
}

impl<'a, const ORDER: usize> RK4Solver<'a, ORDER> {
  fn phi_bar_n(&mut self, n: usize, phi_idx: usize, x: &[f64], xdot: &[f64]) -> (f64, f64) {
    if n == 0 {
      return (xdot[phi_idx], x[phi_idx]);
    }

    let mut phi_bar_dot = xdot[phi_idx];
    let k_row = &self.k[phi_idx];
    for m in 0..n {
      phi_bar_dot += self.dt * self.a[n][m] * k_row[m];
    }

    let phi_bar = x[phi_idx] + self.dt * phi_bar_dot;

    (phi_bar_dot, phi_bar)
  }

  fn h_bar_n(&mut self, n: usize, hubble: f64) -> f64 {
    // eprintln!("Computing hbar{n}");
    if n == 0 {
      return hubble;
    }
    let mut h_bar = hubble;
    for m in 0..n {
      h_bar += self.dt * self.a[n][m] * self.l[m];
    }
    h_bar
  }

  fn update_kn_ln(&mut self, n: usize, x: &[f64], xdot: &[f64], hubble: f64) {
    self.scratch1.fill(0.);
    self.scratch2.fill(0.);

    for a in 0..self.scratch2.len() {
      unsafe {
        (*self.scratch2.get_unchecked_mut(a), *self.scratch1.get_unchecked_mut(a)) =
          self.phi_bar_n(n, a, x, xdot);
      }
      self.phi_dot[a][n] = self.scratch2[a];
    }

    let hubble_bar = self.h_bar_n(n, hubble);

    for (idx, k) in self.k.iter_mut().enumerate() {
      k[n] = unsafe { self.eom.f(idx, &self.scratch1, &self.scratch2, hubble_bar) };
    }

    self.l[n] = unsafe { self.eom.g(&self.scratch1, hubble_bar) };

  }

  pub fn step_rk4(&mut self, x: &mut [f64], xdot: &mut [f64], hubble: &mut f64) -> bool {
    // First compute all the ki's
    self.k.iter_mut().for_each(|v| v.fill(0.));
    self.phi_dot.iter_mut().for_each(|v| v.fill(0.));
    self.l.fill(0.);
    (0..4).for_each(|n| self.update_kn_ln(n, x, xdot, *hubble));

    // Compute two candidates for the next step (for error estimation)
    self.scratch1.fill(0.);
    self.scratch2.fill(0.);

    let w = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0];
    for a in 0..self.eom.lib.n_fields() {
        let dphi = w.iter().enumerate().map(|(i,wi)| wi*self.phi_dot[a][i]).sum::<f64>() * self.dt;
        let dpi  = w.iter().enumerate().map(|(i,wi)| wi*self.k[a][i]).sum::<f64>() * self.dt;
        x[a]    += dphi;
        xdot[a] += dpi;
    }
    *hubble += w.iter().enumerate().map(|(i,wi)| wi*self.l[i]).sum::<f64>() * self.dt;
 

    return false;
  }

  

  pub fn new_rk4(eom: EoM<'a>, dt: f64) -> RK4Solver<'a, 4> {
    const A1: [f64; 4] = [0., 0., 0., 0.];
    const A2: [f64; 4] = [0.5, 0., 0., 0.];
    const A3: [f64; 4] = [0., 0.5, 0., 0.];
    const A4: [f64; 4] = [0., 0., 1., 0.];

    RK4Solver {
      k: vec![[0.; 4]; eom.lib.n_fields()],
      phi_dot: vec![[0.; 4]; eom.lib.n_fields()],
      l: [0.; 4],
      a: &[&A1, &A2, &A3, &A4],
      scratch1: vec![0.; eom.lib.n_fields()],
      scratch2: vec![0.; eom.lib.n_fields()],
      eom,
      dt,
    }
  }
}

#[pyfunction]
pub fn solve_eom_rk4(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  dt: f64,
) -> PyResult<()> {
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let lib = &lib.0;
  let eom = EoM::new(lib, &p)?;
  let mut out = out.as_array_mut();
  let mut solver = RK4Solver::<4>::new_rk4(eom, dt);
  let mut previous_step = out.slice(nd::s![0, 0..]).as_slice().unwrap().to_vec();

  // The hubble paramter is as-of-yet undefined. We use the constraint equation to initialise it
  let (hubble, rest) = previous_step.split_last_mut().unwrap();
  let (x, xdot) = rest.split_at_mut(lib.n_fields());
  *hubble = unsafe { (lib.get_hubble_constraint()?)(x.as_ptr(), xdot.as_ptr(), p.as_ptr()) };

  {
    let mut row0_view = out.slice_mut(nd::s![0, ..]);
    let row0 = row0_view.as_slice_mut().unwrap();
    row0.copy_from_slice(&previous_step);
  }


  for mut row in out.axis_iter_mut(nd::Axis(0)).skip(1) {
    // Copy previous row into this one
    let row = row.as_slice_mut().unwrap();
    row.copy_from_slice(&previous_step);

    // Update the next row
    let (hubble, rest) = row.split_last_mut().unwrap();
    let (x, xdot) = rest.split_at_mut(lib.n_fields());
    while solver.step_rk4(x, xdot, hubble) {}

    // Make the previous row the current row
    previous_step.copy_from_slice(row);
  }

  Ok(())
}