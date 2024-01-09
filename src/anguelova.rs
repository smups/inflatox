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
  the European Union along with inflatox.  If not, see
  <https://ec.europa.eu/info/european-union-public-licence_en/>.

  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
  licensee subject to Dutch law as per article 15 of the EUPL.
*/

use std::io::Write;

use indicatif::{ProgressIterator, ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressStyle};
use nd::ArrayView2;
use ndarray as nd;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PySystemError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hesse_bindings::{Hesse2D, InflatoxDylib, Grad};

#[inline]
fn validate<'lib, T>(
  lib: &'lib InflatoxDylib,
  x: ArrayView2<T>,
  p: &[f64],
) -> PyResult<(Hesse2D<'lib>, Grad<'lib>)> {
  //(1) Make sure we have a two field model
  if !lib.get_n_fields() == 2 {
    crate::raise_shape_err(format!(
      "the Anguelova & Lazaroiu consistency condition requires a 2-field model. Model \"{}\" has only {} fields.",
      lib.name(),
      lib.get_n_fields()
    ))?;
  }
  let h = Hesse2D::new(lib);
  let g = Grad::new(lib);

  //(2) Make sure the field-space array is actually 2d
  if x.shape().len() != 2 {
    crate::raise_shape_err(format!(
      "expected a 2D field-space array. Found array with shape {:?}",
      x.shape()
    ))?;
  }

  //(3) Make sure that the number of supplied model parameters matches the number
  //specified by the dynamic lib
  if p.len() != h.get_n_params() {
    crate::raise_shape_err(format!(
      "model \"{}\" expected {} parameters. Parameter array has {}.",
      lib.name(),
      h.get_n_params(),
      p.len()
    ))?;
  }

  Ok((h, g))
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

#[pyfunction]
/// Evaluate the consistency condition ONLY, not considering any additional
/// parameters.
fn consistency_only(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else {rayon::current_num_threads()};

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, out.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating consistency condition ONLY using {num_threads} threads.");
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out.as_slice_mut().expect("[LIBINFLX_RS_PANIC]: OUTPUT ARRAY NOT C-CONTIGUOUS");
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    let (v, v11, v10, v00) = (h.potential(x, p), h.v11(x, p), h.v10(x, p), h.v00(x, p));
    let lhs = v11/v - 3.;
    let rhs = (3. + v00/v) * (v10/v00).powi(2);
    //Return left-hand-side minus right-hand-side
    *val = (lhs - rhs).abs()
  };

  //(5b) setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out.into_iter()
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(op);
    } else {
      iter.for_each(op);
    }
  } else {
    //Multi-threaded mode
    let threadpool = rayon::ThreadPoolBuilder::new()
      .num_threads(num_threads)
      .build()
      .expect("[LIBINFLX_RS_PANIC]: COULD NOT INITIALISE THREADPOOL");
    threadpool.install(move || {
      let iter = out.into_par_iter()
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(op);
      } else {
        iter.for_each(op);
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

#[pyfunction]
/// Evaluate the consistency condition ONLY using the rapid-turn approximations
/// from Anguelova & Lazaroiu's original paper (not considering any additional
/// parameters).
fn consistency_rapidturn_only(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else {rayon::current_num_threads()};

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, out.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating consistency condition ONLY assuming rapid-turn using {num_threads} threads.");
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out.as_slice_mut().expect("[LIBINFLX_RS_PANIC]: OUTPUT ARRAY NOT C-CONTIGUOUS");
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    let (v, v11, v10, v00) = (h.potential(x, p), h.v11(x, p), h.v10(x, p), h.v00(x, p));
    let lhs = v11/v;
    let rhs = 3. * (v10/v00).powi(2);
    //Return left-hand-side minus right-hand-side
    *val = (lhs - rhs).abs()
  };

  //(5b) setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out.into_iter()
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(op);
    } else {
      iter.for_each(op);
    }
  } else {
    //Multi-threaded mode
    let threadpool = rayon::ThreadPoolBuilder::new()
      .num_threads(num_threads)
      .build()
      .expect("[LIBINFLX_RS_PANIC]: COULD NOT INITIALISE THREADPOOL");
    threadpool.install(move || {
      let iter = out.into_par_iter()
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(op);
      } else {
        iter.for_each(op);
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

fn iter_array<'a, T: Send>(
  x: &'a mut [T],
  start_stop: &[[f64; 2]],
  shape: &'a [usize],
) -> impl IndexedParallelIterator<Item = ([f64; 2], &'a mut T)> {
  //(1) Calculate spacings
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(start_stop, shape);

  //(2) Set-up iterator over field-space array
  x.into_par_iter()
    .enumerate()
    //(2a) convert flat index into array index
    .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
    //(2b) convert array index into field-space point
    .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val))
}

fn set_pbar(len: usize) -> ProgressBar {
  const PBAR_REFRESH: u8 = 5;
  const PBAR_STYLE: &str =
    "Time to completion: {eta:<.0}\nOperations/s: {per_sec}\n{bar:40.blue/gray} {percent}%";
  let style = ProgressStyle::default_bar().template(PBAR_STYLE).unwrap();
  let target = ProgressDrawTarget::stderr_with_hz(PBAR_REFRESH);

  ProgressBar::with_draw_target(Some(len as u64), target).with_style(style)
}

fn anguelova_leading_order(
  h: Hesse2D,
  mut x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
  progress: bool,
) {
  let shape = &[x.shape()[0], x.shape()[1]];
  let len = x.len();
  let iter = iter_array(x.as_slice_mut().unwrap(), start_stop, shape);

  //Leading order calculation as closure
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    *val = {
      let lhs = 3.0 * (h.v00(x, p) / h.v01(x, p)).powi(2);
      let rhs = h.v11(x, p) / h.potential(x, p);
      ((lhs / rhs) - 1.0).abs()
    }
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }
}

fn anguelova_0th_order(
  h: Hesse2D,
  mut x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
  progress: bool,
) {
  let shape = &[x.shape()[0], x.shape()[1]];
  let len = x.len();
  let iter = iter_array(x.as_slice_mut().unwrap(), start_stop, shape);

  //Zeroth order calculation as closure
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    *val = {
      let lhs = 3.0 * (h.v00(x, p) / h.v01(x, p)).powi(2) + 1.0; //the +1.0 is the zeroth order correction
      let rhs = h.v11(x, p) / h.potential(x, p);
      ((lhs / rhs) - 1.0).abs()
    }
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }
}

fn anguelova_2nd_order(
  h: Hesse2D,
  mut x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
  progress: bool,
) {
  let shape = &[x.shape()[0], x.shape()[1]];
  let len = x.len();
  let iter = iter_array(x.as_slice_mut().unwrap(), start_stop, shape);

  //Second order calculation as closure
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    *val = {
      let (v, v00, v10, v11) = (h.potential(x, p), h.v00(x, p), h.v10(x, p), h.v11(x, p));
      let lhs = 3.0 * (v00 / v10).powi(2) + v10.powi(2) / (v * v00) + 0.2 * (v10 / v00).powi(2);
      let rhs = v11 / v - 1.0;
      ((lhs / rhs) - 1.0).abs()
    }
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }
}

fn anguelova_exact(
  h: Hesse2D,
  mut x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]],
  progress: bool,
) {
  let shape = &[x.shape()[0], x.shape()[1]];
  let len = x.len();
  let iter = iter_array(x.as_slice_mut().unwrap(), start_stop, shape);

  //Exact calculation as closure
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    *val = {
      let (v, v00, v10, v11) = (h.potential(x, p), h.v00(x, p), h.v10(x, p), h.v11(x, p));
      let tan2_delta = (v10 / v00).powi(2);
      let csc2_delta = 1.0 + tan2_delta.recip(); //csc²(x) = 1 + cot²(x)
      let lhs = 3.0 * csc2_delta + (v00 / v) * tan2_delta;
      let rhs = v11 / v;
      ((lhs / rhs) - 1.0).abs()
    }
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }
}

#[pyfunction]
/// python-facing function that evaluates Anguelova & Lazaroiu's consistency
/// condition for a two-field model for the supplied input field-space array x
/// and the parameter array p. The order of the calculation may be specified using
/// the order parameter. Console output will be generated if progress=true.
pub fn anguelova_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  order: isize,
  progress: bool,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating consistency condition using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) evaluate anguelova's condition up to the specified order
  match order {
    o if o < -1 => anguelova_exact(h, x, p, &start_stop, progress),
    -1 => anguelova_leading_order(h, x, p, &start_stop, progress),
    0 => anguelova_0th_order(h, x, p, &start_stop, progress),
    2 => anguelova_2nd_order(h, x, p, &start_stop, progress),
    o => {
      return Err(PySystemError::new_err(format!(
        "expected order to be -1, 0, 2 or smaller than -1. Found {o}"
      )))
    }
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );
  Ok(())
}

#[pyfunction]
/// python-facing function used to calculate the characteristic angle delta given
/// the supplied input field-space array and the parameter array p. The order of
/// the calculation may be specified using the order parameter. Console output
/// will be generated if progress=true.
pub fn delta_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating alignment angle δ using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = |(ref x, val): ([f64; 2], &mut f64)| *val = (h.v01(x, p) / h.v00(x, p)).atan();

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

#[pyfunction]
/// python-facing function used to calculate the relative turn rate omega given
/// the supplied input field-space array and the parameter array p. The order of
/// the calculation may be specified using the order parameter. Console output
/// will be generated if progress=true.
pub fn omega_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating turn rate ω using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    let (v, v00, v01, v11) = (h.potential(x, p), h.v00(x, p), h.v01(x, p), h.v11(x, p));
    let cos2d = v00.powi(2) / (v00.powi(2) + v01.powi(2));
    let sin2d = v01.powi(2) / (v00.powi(2) + v01.powi(2));
    let sincosd = (v01 * v00) / (v00.powi(2) + v01.powi(2));
    let vtt = cos2d * v11 + sin2d * v00 - 2.0 * sincosd * v01;
    *val = (3.0 * vtt / v).abs().sqrt();
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

#[pyfunction]
/// python-facing function used to calculate the first slow-roll parameter
/// epsilon given the supplied input field-space array and the parameter array p.
/// Console output will be generated if progress=true.
pub fn epsilon_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, grad) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating first slow-roll paramter ε using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    //Calculate omega
    let (v, v00, v01, v11) = (h.potential(x, p), h.v00(x, p), h.v01(x, p), h.v11(x, p));
    let cos2d = v00.powi(2) / (v00.powi(2) + v01.powi(2));
    let sin2d = v01.powi(2) / (v00.powi(2) + v01.powi(2));
    let sincosd = (v01 * v00) / (v00.powi(2) + v01.powi(2));
    let vtt_byv = cos2d * (v11/v) + sin2d * (v00/v) - 2.0 * sincosd * (v01/v);
    let omega2_by9 = (vtt_byv / 3.0).abs();

    //Calculate epsilon_V
    let epsilon_v = grad.grad_square(x, p) / (2.0 * v.powi(2));

    //Calculate epsilon
    *val = epsilon_v / (1. + omega2_by9);
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

#[pyfunction]
/// python-facing function used to calculate the second slow-roll parameter
/// eta given the supplied input field-space array and the parameter array p.
/// Console output will be generated if progress=true.
pub fn eta_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, _) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating second slow-roll parameter η using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    //Calculate omega
    let (v, v00, v01, v11) = (h.potential(x, p), h.v00(x, p), h.v01(x, p), h.v11(x, p));
    let cos2d = v00.powi(2) / (v00.powi(2) + v01.powi(2));
    let sin2d = v01.powi(2) / (v00.powi(2) + v01.powi(2));
    let sincosd = (v01 * v00) / (v00.powi(2) + v01.powi(2));
    let vtt_byv = cos2d * (v11/v) + sin2d * (v00/v) - 2.0 * sincosd * (v01/v);
    let omega = (vtt_byv * 3.0).abs().sqrt();

    //Calculate tan(delta)
    let tandelta = h.v01(x, p) / h.v00(x, p);
    *val = (3. - (omega * tandelta).abs()).abs()
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}

#[pyfunction]
/// python-facing function that produces a masked array indicating which pixels
/// may induce a sign-flip of the gradient of V. This function flags those pixels
/// where the components of the gradient become very small, where very small is
/// defined by the user-supplied value `accuracy`.
pub fn flag_quantum_dif_py(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<bool>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  accuracy: f64
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect("[LIBINFLX_RS_PANIC]: PARAMETER ARRAY NOT C-CONTIGUOUS");
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (_, g) = validate(lib, x.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating zeros of the potential gradient using {} threads.", rayon::current_num_threads());
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = |(ref x, val): ([f64; 2], &mut bool)| {
    *val = (g.cmp(x, p, 0).abs() <= accuracy) & (g.cmp(x, p, 1).abs() <= accuracy);
  };

  if progress {
    iter.progress_with(set_pbar(len)).for_each(op);
  } else {
    iter.for_each(op);
  }

  //(6) Report how long we took, and return.
  eprintln!(
    "[Inflatox] Calculation finished. Took {}.",
    indicatif::HumanDuration(start.elapsed()).to_string()
  );

  Ok(())
}
