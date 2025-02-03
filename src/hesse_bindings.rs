/*
  Copyright© 2023 Raúl Wolters(*)

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

  (*) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/
use ndarray as nd;

use crate::dylib::{ExFn2, ExVecFn, InflatoxDylib};
use crate::BADGE_PANIC;

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

pub struct Potential<'a> {
  lib: &'a InflatoxDylib,
  potential: ExFn2,
  grad: ExVecFn,
  grad_square: ExFn2,
}

impl<'a> Potential<'a> {
  #[inline(always)]
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self> {
    Ok(Potential {
      lib,
      potential: lib.potential(),
      // 0th basis function is always the normalised gradient
      grad: lib.get_basis_fn(0)?,
      grad_square: lib.grad_square(),
    })
  }

  #[inline(always)]
  /// Compute the potential.
  ///
  /// # Panics
  /// Panics the length of ` x` is smaller than the number of fields or if the lenght of `p` is
  /// smaller than the number of parameters.
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    unsafe { (self.potential)(x.as_ptr(), p.as_ptr()) }
  }

  /// Calculate scalar potential all field-space coordinates in the array `x`,
  /// with model parameters `p`. The physical range that the elements in `x`
  /// represent can be specified by passing the start_stop array.
  ///
  /// # Panics
  /// This function panics if `x.shape().len()` does not equal the number of
  /// fields of the loaded model. Similarly, if `p.len()` does not equal the
  /// number of model parameters, this function will panic.
  pub fn potential_array(&self, mut x: nd::ArrayViewMutD<f64>, p: &[f64], start_stop: &[[f64; 2]]) {
    assert!(x.shape().len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    // Convert start-stop ranges
    let (spacings, offsets) = start_stop
      .iter()
      .zip(x.shape().iter())
      .map(|([start, stop], &axis_len)| ((stop - start) / axis_len as f64, *start))
      .unzip::<_, _, Vec<_>, Vec<_>>();
    let mut field_space_point = Vec::with_capacity(x.shape().len());

    for (idx, val) in x.indexed_iter_mut() {
      field_space_point.clear();
      field_space_point
        .extend((0..self.lib.n_fields()).map(|i| idx[i] as f64 * spacings[i] + offsets[i]));
      *val = unsafe { (self.potential)(field_space_point.as_ptr(), p.as_ptr()) };
    }
  }

  #[inline(always)]
  /// Compute the `idx`'th component of the gradient of the potential.
  ///
  /// # Panics
  /// Panics the length of ` x` is smaller than the number of fields or if the lenght of `p` is
  /// smaller than the number of parameters.
  pub fn grad(&self, x: &[f64], p: &[f64], out: &mut [f64]) {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    assert!(out.len() <= self.lib.n_fields(), "{}", *BADGE_PANIC);
    unsafe { (self.grad)(x.as_ptr(), p.as_ptr(), out.as_mut_ptr()) };
  }

  #[inline(always)]
  /// Compute the square of the gradient of the potential.
  ///
  /// # Panics
  /// Panics the length of ` x` is smaller than the number of fields or if the lenght of `p` is
  /// smaller than the number of parameters.
  pub fn grad_square(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    unsafe { (self.grad_square)(x.as_ptr(), p.as_ptr()) }
  }
}

pub struct Hesse<'a> {
  lib: &'a InflatoxDylib,
  hesse: nd::Array2<ExFn2>,
}

impl<'a> Hesse<'a> {
  #[inline(always)]
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self> {
    Ok(Hesse { lib, hesse: lib.get_hesse_cmp()? })
  }

  #[inline(always)]
  /// Calculate projected Hesse matrix at field-space coordinates `x`, with model
  /// parameters `p`.
  ///
  /// # Panics
  /// This function panics if `x.len()` does not equal the number of fields of
  /// the loaded model. Similarly, if `p.len()` does not equal the number of
  /// model parameters, this function will panic.
  pub fn hesse(&self, x: &[f64], p: &[f64]) -> nd::Array2<f64> {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    self.hesse.mapv(|func| unsafe { func(x.as_ptr(), p.as_ptr()) })
  }

  /// Calculate the projected hesse matrix at a number of field-space coordinates.
  /// The `x_shape` parameter determines the number of samples along each axis.
  /// The `start_stop` arrays indicate the values of the first and last samples
  /// along each axis. The model parameters must be supplied via the slice `p`.
  /// Returns an array with two more axes than `x_shape` has entries,
  /// representing the axes of the hesse matrix. The FIRST two axes of the output
  /// array represent the axes of the hesse array.
  ///
  /// # Panics
  /// This function panics if `x.len()` does not equal the number of fields of
  /// the loaded model. Similarly, if `p.len()` does not equal the number of
  /// model parameters, this function will panic.
  pub fn hesse_array(
    &self,
    x_shape: &[usize],
    p: &[f64],
    start_stop: &[[f64; 2]],
  ) -> nd::ArrayD<f64> {
    let n_fields = self.lib.n_fields();
    assert!(x_shape.len() == n_fields, "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);

    // Convert start-stop ranges
    let (spacings, offsets) = start_stop
      .iter()
      .zip(x_shape.iter())
      .map(|([start, stop], &axis_len)| ((stop - start) / axis_len as f64, *start))
      .unzip::<_, _, Vec<_>, Vec<_>>();

    // Create output array
    let output_shape = [vec![n_fields, n_fields], x_shape.to_vec()].concat();
    let mut output = nd::ArrayD::<f64>::zeros(output_shape);

    // Fill output array
    let mut field_space_point = Vec::with_capacity(x_shape.len());
    output.axis_iter_mut(nd::Axis(0)).enumerate().for_each(|(i, mut view)| {
      view.axis_iter_mut(nd::Axis(0)).enumerate().for_each(|(j, mut x)| {
        // Just to be clear, the first two axes are the axes of the hesse
        // matrix. All the other axes are the field-space axes (we do not
        // know how many of these there are). We will thus be calculating the
        // ijth component of the hesse matrix for ALL field space points x,
        // and then moving on to ij+1 etc...
        x.indexed_iter_mut().for_each(|(idx, val)| {
          // Convert index into field_space point
          field_space_point.clear();
          field_space_point.extend((0..n_fields).map(|k| idx[k] as f64 * spacings[k] + offsets[k]));
          // Calculate the ijth matrix element
          let x_ptr = field_space_point.as_ptr();
          let p_ptr = p.as_ptr();
          *val = unsafe { (self.hesse[(i, j)])(x_ptr, p_ptr) };
        })
      });
    });
    output
  }
}

pub struct Hesse2D<'a> {
  lib: &'a InflatoxDylib,
  fns: [ExFn2; 4],
}

impl<'a> Hesse2D<'a> {
  #[inline(always)]
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self> {
    assert!(lib.n_fields() == 2);
    let hesse = lib.get_hesse_cmp()?;
    let v00 = hesse[(0, 0)];
    let v01 = hesse[(0, 1)];
    let v10 = hesse[(1, 0)];
    let v11 = hesse[(1, 1)];
    Ok(Hesse2D { lib, fns: [v00, v01, v10, v11] })
  }

  #[inline(always)]
  pub fn v00(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    unsafe { self.fns[0](x.as_ptr(), p.as_ptr()) }
  }

  #[inline(always)]
  pub fn v10(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    unsafe { self.fns[2](x.as_ptr(), p.as_ptr()) }
  }

  #[inline(always)]
  pub fn v11(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.n_fields(), "{}", *BADGE_PANIC);
    assert!(p.len() == self.lib.n_pars(), "{}", *BADGE_PANIC);
    unsafe { self.fns[3](x.as_ptr(), p.as_ptr()) }
  }
}
