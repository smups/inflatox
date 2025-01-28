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

struct RKNSolver<'a, const ORDER: usize> {
  eom: EoM<'a>,
  dt: f64,
  max_err: f64,
  phi_err: Vec<f64>,
  phidot_err: Vec<f64>,
  a: &'static [&'static [f64]; ORDER],
  b: &'static [f64; ORDER],
  bbar: &'static [f64; ORDER],
  c: &'static [f64; ORDER],
  k: Vec<[f64; ORDER]>,
  l: [f64; ORDER],
  scratch1: Vec<f64>,
  scratch2: Vec<f64>,
  scratch3: Vec<f64>,
  scratch4: Vec<f64>,
}

impl<'a, const ORDER: usize> RKNSolver<'a, ORDER> {
  fn phi_bar_n(&mut self, n: usize, phi_idx: usize, x: &[f64], xdot: &[f64]) -> (f64, f64) {
    if n == 0 {
      return (xdot[phi_idx], x[phi_idx]);
    }

    let k = &self.k[phi_idx];
    let mut phi_bar_dot = xdot[phi_idx];
    let mut phi_bar = x[phi_idx];
    for m in 0..n - 1 {
      phi_bar_dot += self.dt * self.a[n][m] * k[m];
      phi_bar += self.dt * self.c[m] * xdot[phi_idx];
      for o in 0..n - 1 {
        phi_bar += self.dt.powi(2) * self.a[m][o] * k[o];
      }
    }

    (phi_bar_dot, phi_bar)
  }

  fn h_bar_n(&mut self, n: usize, hubble: f64) -> f64 {
    // eprintln!("Computing hbar{n}");
    if n == 0 {
      return hubble;
    }
    let mut h_bar = hubble;
    for m in 0..n - 1 {
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
    }
    let hubble_bar = self.h_bar_n(n, hubble);

    for (idx, k) in self.k.iter_mut().enumerate() {
      k[n] = unsafe { self.eom.f(idx, &self.scratch2, &self.scratch1, hubble_bar) };
    }

    for li in &mut self.l {
      *li = unsafe { self.eom.g(&self.scratch2, hubble) };
    }
  }

  pub fn step(&mut self, x: &mut [f64], xdot: &mut [f64], hubble: &mut f64) -> bool {
    // First compute all the ki's
    self.k.iter_mut().for_each(|v| v.fill(0.));
    self.l.fill(0.);
    (0..ORDER).for_each(|n| self.update_kn_ln(n, x, xdot, *hubble));

    // Compute two candidates for the next step (for error estimation)
    self.scratch1.fill(0.);
    self.scratch2.fill(0.);
    self.scratch3.fill(0.);
    self.scratch4.fill(0.);
    for a in 0..self.eom.lib.n_fields() {
      self.scratch2[a] = self.dt * xdot[a]
        + self.dt.powi(2)
          * self.k[a].iter().zip(self.b.iter()).fold(0., |acc, (ki, bi)| acc + bi + ki);
      self.scratch3[a] = self.dt * xdot[a]
        + self.dt.powi(2)
          * self.k[a].iter().zip(self.bbar.iter()).fold(0., |acc, (ki, bi)| acc + bi + ki);
      self.scratch1[a] =
        self.dt * self.k[a].iter().zip(self.b.iter()).fold(0.0, |acc, (ki, bi)| acc + ki * bi);
      self.scratch4[a] =
        self.dt * self.k[a].iter().zip(self.bbar.iter()).fold(0.0, |acc, (ki, bi)| acc + ki * bi);
      self.phi_err[a] = (self.scratch2[a] - self.scratch3[a]).abs();
      self.phidot_err[a] = (self.scratch1[a] - self.scratch4[a]).abs();
    }
    let d_hubble1 =
      self.dt * self.l.iter().zip(self.b.iter()).fold(0.0, |acc, (li, bi)| acc + li * bi);
    let d_hubble2 =
      self.dt * self.l.iter().zip(self.bbar.iter()).fold(0.0, |acc, (li, bi)| acc + li * bi);

    let hubble_err = (d_hubble2 - d_hubble1).abs();
    let phi_err = self.phi_err.iter().fold(0., |acc, err| acc + err.powi(2)).sqrt();
    let phidot_err = self.phidot_err.iter().fold(0., |acc, err| acc + err.powi(2)).sqrt();
    let err = (hubble_err.powi(2) + phi_err.powi(2) + phidot_err.powi(2)).sqrt();

    if err / self.max_err > 1.1 {
      let q = (self.max_err / err).powf(((ORDER + 1) as f64).recip());
      self.dt *= q;
      assert!(q < 1.);
      return true;
    } else if err / self.max_err < 1. {
      let q = (self.max_err / err).powf(((ORDER + 1) as f64).recip());
      self.dt *= q;
      assert!(q > 1.);
    }

    // Step hubble parameter
    *hubble += d_hubble2;

    // Step fields and their derivatives
    for a in 0..self.eom.lib.n_fields() {
      unsafe {
        *x.get_unchecked_mut(a) += self.scratch3.get_unchecked(a);
        *xdot.get_unchecked_mut(a) += self.scratch4.get_unchecked(a);
      }
    }

    return false;
  }

  pub fn new_rk4(eom: EoM<'a>, max_err: f64) -> RKNSolver<'a, 4> {
    const A1: [f64; 4] = [0., 0., 0., 0.];
    const A2: [f64; 4] = [0.5, 0., 0., 0.];
    const A3: [f64; 4] = [0., 0.5, 0., 0.];
    const A4: [f64; 4] = [0., 0., 1., 0.];
    const B: [f64; 4] = [1. / 6., 1. / 3., 1. / 3., 1. / 6.];
    const BBAR: [f64; 4] = B;
    const C: [f64; 4] = [0., 0.5, 0.5, 1.];

    RKNSolver {
      k: vec![[0.; 4]; eom.lib.n_fields()],
      l: [0.; 4],
      a: &[&A1, &A2, &A3, &A4],
      b: &B,
      bbar: &BBAR,
      c: &C,
      scratch1: vec![0.; eom.lib.n_fields()],
      scratch2: vec![0.; eom.lib.n_fields()],
      scratch3: vec![0.; eom.lib.n_fields()],
      scratch4: vec![0.; eom.lib.n_fields()],
      phi_err: vec![0.; eom.lib.n_fields()],
      phidot_err: vec![0.; eom.lib.n_fields()],
      eom,
      dt: 1e-10,
      max_err,
    }
  }

  pub fn new_rkf(eom: EoM<'a>, max_err: f64) -> RKNSolver<'a, 6> {
    const A1: [f64; 5] = [0.; 5];
    const A2: [f64; 5] = [0.25, 0., 0., 0., 0.];
    const A3: [f64; 5] = [3. / 32., 9. / 32., 0., 0., 0.];
    const A4: [f64; 5] = [1932. / 2197., -7200. / 2197., 7296. / 2197., 0., 0.];
    const A5: [f64; 5] = [439. / 216., -8., 3680. / 513., -845. / 4104., 0.];
    const A6: [f64; 5] = [-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.];
    const B1: [f64; 6] = [16. / 135., 0., 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.];
    const B2: [f64; 6] = [25. / 216., 0., 1408. / 2565., 2197. / 4104., -1. / 5., 0.];
    const C: [f64; 6] = [0., 0.25, 3. / 8., 12. / 13., 1., 0.5];

    RKNSolver {
      k: vec![[0.; 6]; eom.lib.n_fields()],
      l: [0.; 6],
      a: &[&A1, &A2, &A3, &A4, &A5, &A6],
      b: &B1,
      bbar: &B2,
      c: &C,
      scratch1: vec![0.; eom.lib.n_fields()],
      scratch2: vec![0.; eom.lib.n_fields()],
      scratch3: vec![0.; eom.lib.n_fields()],
      scratch4: vec![0.; eom.lib.n_fields()],
      phi_err: vec![0.; eom.lib.n_fields()],
      phidot_err: vec![0.; eom.lib.n_fields()],
      eom,
      dt: 1e-10,
      max_err,
    }
  }
}

#[pyfunction]
pub fn solve_eom_rk4(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  max_err: f64,
) -> PyResult<()> {
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let lib = &lib.0;
  let eom = EoM::new(lib, &p)?;
  let mut out = out.as_array_mut();
  let mut solver = RKNSolver::<0>::new_rk4(eom, max_err);
  let mut previous_step = out.slice(nd::s![0, 0..]).as_slice().unwrap().to_vec();

  // The hubble paramter is as-of-yet undefined. We use the constraint equation to initialise it
  let (hubble, rest) = previous_step.split_last_mut().unwrap();
  let (x, xdot) = rest.split_at_mut(lib.n_fields());
  *hubble = unsafe { (lib.get_hubble_constraint()?)(x.as_ptr(), xdot.as_ptr(), p.as_ptr()) };

  for mut row in out.axis_iter_mut(nd::Axis(0)).into_iter() {
    // Copy previous row into this one
    let row = row.as_slice_mut().unwrap();
    row.copy_from_slice(&previous_step);

    // Update the next row
    let (hubble, rest) = row.split_last_mut().unwrap();
    let (x, xdot) = rest.split_at_mut(lib.n_fields());
    while solver.step(x, xdot, hubble) {}

    // Make the previous row the current row
    previous_step.copy_from_slice(row);
  }

  Ok(())
}

#[pyfunction]
pub fn solve_eom_rkf(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  max_err: f64,
) -> PyResult<()> {
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let lib = &lib.0;
  let eom = EoM::new(lib, &p)?;
  let mut out = out.as_array_mut();
  let mut solver = RKNSolver::<0>::new_rkf(eom, max_err);
  let mut previous_step = out.slice(nd::s![0, 0..]).as_slice().unwrap().to_vec();

  // The hubble paramter is as-of-yet undefined. We use the constraint equation to initialise it
  let (hubble, rest) = previous_step.split_last_mut().unwrap();
  let (x, xdot) = rest.split_at_mut(lib.n_fields());
  *hubble = unsafe { (lib.get_hubble_constraint()?)(x.as_ptr(), xdot.as_ptr(), p.as_ptr()) };

  for mut row in out.axis_iter_mut(nd::Axis(0)).into_iter() {
    // Copy previous row into this one
    let row = row.as_slice_mut().unwrap();
    row.copy_from_slice(&previous_step);

    // Update the next row
    let (hubble, rest) = row.split_last_mut().unwrap();
    let (x, xdot) = rest.split_at_mut(lib.n_fields());
    while solver.step(x, xdot, hubble) {}

    // Make the previous row the current row
    previous_step.copy_from_slice(row);
  }

  Ok(())
}
