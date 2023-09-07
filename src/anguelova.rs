/*
  Copyright© 2023 Raúl Wolters(1)

  This file is part of libinflx_rs (rust bindings for inflatox).

  inflatox is free software: you can redistribute it and/or modify it under
  the terms of the European Union Public License version 1.2 or later, as
  published by the European Commission.

  inflatox is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
  A PARTICULAR PURPOSE. See the European Union Public License for more details.

  You should have received a copy of the EUPL in an/all official language(s) of
  the European Union along with inflatox.  If not, see
  <https://ec.europa.eu/info/european-union-public-licence_en/>.

  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/

use ndarray as nd;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PySystemError;
use rayon::prelude::*;

use crate::hesse_bindings::Hesse2D;

#[cfg(feature = "pyo3_extension_module")]
#[pyfunction]
pub(crate) fn anguelova_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  order: isize
) -> PyResult<()> {
  //(0) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_array();
  let x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(1) Make sure we have a two field model
  if !lib.get_n_fields() == 2 {
    crate::raise_shape_err(format!(
      "the Anguelova consistency condition requires a 2-field model. Received a {}-field model.",
      lib.get_n_fields()
    ))?;
  }
  let h = Hesse2D::new(lib);

  //(2) Make sure the field-space array is actually 2d
  if x.shape().len() != 2 {
    crate::raise_shape_err(format!(
      "expected a 2D field-space array. Found array with shape {:?}",
      x.shape()
    ))?;
  }

  //(3) Make sure that the number of supplied model parameters matches the number
  //specified by the dynamic lib
  if p.shape() != &[h.get_n_params()] {
    crate::raise_shape_err(format!(
      "model expected {} parameters, got {}",
      h.get_n_params(),
      p.shape().len()
    ))?;
  }
  let p = p.as_slice().unwrap();

  //(4) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(5) evaluate anguelova's condition up to the specified order
  match order {
    o if o < -1 => anguelova_exact(h, x, p, &start_stop),
    -1 => anguelova_leading_order(h, x, p, &start_stop),
    0 => anguelova_0th_order(h, x, p, &start_stop),
    2 => anguelova_2nd_order(h, x, p, &start_stop),
    o => return Err(PySystemError::new_err(format!("expected order to be negative, 0, 1 or 2. Found {o}")))
  }

  Ok(())
}

/// Converts start-stop ranges into offset-spacing ranges. Order of return arguments
/// is x_spacing, y_spacing, x_ofst, y_ofst
fn convert_ranges(start_stop: &[[f64; 2]], shape: &[usize]) -> (f64, f64, f64, f64) {
  let x_start = start_stop[0][0];
  let x_stop = start_stop[0][1];
  let x_spacing = (x_stop - x_start) / shape[0] as f64;

  let y_start = start_stop[1][0];
  let y_stop = start_stop[1][1];
  let y_spacing = (y_stop - y_start) / shape[1] as f64;

  (x_spacing, y_spacing, x_start, y_start)
}

pub fn anguelova_leading_order(
  h: Hesse2D,
  x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
) {
  //(1) Convert start-stop ranges
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(start_stop, x.shape());

  //(2) Fill output array
  nd::Zip::indexed(x)
    .into_par_iter()
    //(2a) Convert indices to field-space coordinates
    .map(|(idx, val)| ([idx.0 as f64 * x_spacing + x_ofst, idx.1 as f64 * y_spacing + y_ofst], val))
    //(2b) evaluate consistency condition at every field-space point
    .for_each(|(ref x, val)| {
      *val = {
        let lhs = 3.0 * (h.v00(x, p) / h.v01(x, p)).powi(2);
        let rhs = h.v11(x, p) / h.potential(x, p);
        ((lhs / rhs) - 1.0).abs()
      }
    });
}

pub fn anguelova_0th_order(
  h: Hesse2D,
  x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
) {
  //(1) Convert start-stop ranges
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(start_stop, x.shape());

  //(2) Fill output array
  nd::Zip::indexed(x)
    .into_par_iter()
    //(2a) Convert indices to field-space coordinates
    .map(|(idx, val)| ([idx.0 as f64 * x_spacing + x_ofst, idx.1 as f64 * y_spacing + y_ofst], val))
    //(2b) evaluate consistency condition at every field-space point
    .for_each(|(ref x, val)| {
      *val = {
        let lhs = 3.0 * (h.v00(x, p) / h.v01(x, p)).powi(2) + 1.0;
        let rhs = h.v11(x, p) / h.potential(x, p);
        ((lhs / rhs) - 1.0).abs()
      }
    });
}

pub fn anguelova_2nd_order(
  h: Hesse2D,
  x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
) {
  //(1) Convert start-stop ranges
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(start_stop, x.shape());

  //(2) Fill output array
  nd::Zip::indexed(x)
    .into_par_iter()
    //(2a) Convert indices to field-space coordinates
    .map(|(idx, val)| ([idx.0 as f64 * x_spacing + x_ofst, idx.1 as f64 * y_spacing + y_ofst], val))
    //(2b) evaluate consistency condition at every field-space point
    .for_each(|(ref x, val)| {
      *val = {
        let (v, v00, v10, v11) = (h.potential(x, p), h.v00(x, p), h.v10(x, p), h.v11(x, p));
        let lhs = 3.0 * (v00 / v10).powi(2) + v10.powi(2) / (v * v00) + 0.2 * (v10 / v00).powi(2);
        let rhs = v11 / v - 1.0;
        ((lhs / rhs) - 1.0).abs()
      }
    });
}

pub fn anguelova_exact(h: Hesse2D, x: nd::ArrayViewMut2<f64>, p: &[f64], start_stop: &[[f64; 2]]) {
  //(1) Convert start-stop ranges
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(start_stop, x.shape());

  //(2) Fill output array
  nd::Zip::indexed(x)
    .into_par_iter()
    //(2a) Convert indices to field-space coordinates
    .map(|(idx, val)| ([idx.0 as f64 * x_spacing + x_ofst, idx.1 as f64 * y_spacing + y_ofst], val))
    //(2b) evaluate consistency condition at every field-space point
    .for_each(|(ref x, val)| {
      *val = {
        let (v, v00, v10, v11) = (h.potential(x, p), h.v00(x, p), h.v10(x, p), h.v11(x, p));
        let delta = (v10 / v00).atan();
        let lhs = 3.0 * delta.sin().powi(-2) + v10.powi(2) / (v * v00);
        let rhs = v11 / v;
        ((lhs / rhs) - 1.0).abs()
      }
    });
}
