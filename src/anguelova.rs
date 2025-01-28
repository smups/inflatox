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

use indicatif::{
  ParallelProgressIterator, ProgressBar, ProgressDrawTarget, ProgressIterator, ProgressStyle,
};
use numpy::{
  PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{
  dylib::InflatoxDylib,
  hesse_bindings::{Hesse2D, Potential},
  BADGE_INFO, BADGE_PANIC,
};

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

fn set_pbar(len: usize) -> ProgressBar {
  const PBAR_REFRESH: u8 = 2;
  const PBAR_STYLE: &str =
    "Time to completion: {eta:<.0}\nOperations/s: {per_sec}\n{bar:40.magenta/gray} {percent}%";
  let style = ProgressStyle::default_bar().template(PBAR_STYLE).unwrap();
  let target = ProgressDrawTarget::stderr_with_hz(PBAR_REFRESH);

  ProgressBar::with_draw_target(Some(len as u64), target).with_style(style)
}

#[inline]
/// Parses `lib` into `Hesse2D` and `Grad` structs if possible. Returns an error (read: model is
/// incompatible with the AL consistency condition) otherwise.
fn validate_lib(lib: &InflatoxDylib) -> Result<(Hesse2D<'_>, Potential<'_>)> {
  // The AL condition only works for 2-field models.
  if !lib.n_fields() == 2 {
    return Err(Error::Shape {
      expected: vec![2],
      got: vec![lib.n_fields()],
      msg: "the Anguelova & Lazaroiu consistency condition requires a 2-field model.".to_string(),
    });
  }
  Ok((Hesse2D::new(lib)?, Potential::new(lib)?))
}

#[inline]
/// Checks if the length of the paramter array `p` equals the number of paramters expected by the
/// model
fn validiate_p(lib: &InflatoxDylib, p: &[f64]) -> Result<()> {
  if p.len() != lib.n_pars() {
    return Err(Error::Shape {
      expected: vec![2],
      got: vec![p.len()],
      msg: format!("model \"{}\" has {} paramters", lib.name(), lib.n_pars()),
    });
  }
  Ok(())
}

#[inline(always)]
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

/// Module containing the actual implementations of all the calculations performed by inflatox.
/// These are used in various `#[pyfunction]`s and listed here to make sure they all use exactly
/// the same expressions. All these functions should be marked as `#[inline(always)]`.
mod ops {
  use super::*;

  #[inline(always)]
  pub fn complete_analysis(
    x: [f64; 2],
    val: &mut [f64],
    h: &Hesse2D<'_>,
    g: &Potential<'_>,
    p: &[f64],
  ) {
    let (v, v11, v10, v00) = (g.potential(&x, p), h.v11(&x, p), h.v10(&x, p), h.v00(&x, p));
    //(1) Calculate consistency condition
    let consistency = {
      let lhs = v11 / v;
      let rhs = 3. + 3. * (v00 / v10).powi(2) + (v00 / v) * (v10 / v00).powi(2);
      (lhs - rhs).abs() / (lhs.abs() + rhs.abs())
      //(lhs - rhs).abs()
    };
    // Calculate ε_V
    let epsilon_v = g.grad_square(&x, p) / v.powi(2);
    // Calculate Vtt
    let vtt = (v00 * v10.powi(2) + v11 * v00.powi(2) - 2. * v00 * v10.powi(2))
      / (v00.powi(2) + v10.powi(2));
    // Calculate (Vt)²
    let vt2 = epsilon_v * (1. + (v00 / v10).powi(2)).recip();
    // Calculate ε_H
    let epsilon_h = 3. * (epsilon_v - vt2) * (epsilon_v + vtt.abs() / v - vt2).recip();
    // Calculate δ
    let delta = (v10 / v00).abs().atan();
    // Calculate ω
    let omega = ((vtt / v) * (3. - epsilon_h)).sqrt();
    // Calculate η_H
    let eta_parallel = omega * delta.tan() - 3.;

    val.swap_with_slice(&mut [consistency, epsilon_v, epsilon_h, eta_parallel, delta, omega]);
  }

  #[inline(always)]
  pub fn epsilon_v_only(x: [f64; 2], p: &[f64], _: &Hesse2D<'_>, g: &Potential<'_>) -> f64 {
    0.5 * g.grad_square(&x, p) / g.potential(&x, p).powi(2)
  }

  #[inline(always)]
  pub fn consistency_rapidturn_only(
    x: [f64; 2],
    p: &[f64],
    h: &Hesse2D<'_>,
    g: &Potential<'_>,
  ) -> f64 {
    let (v, v11, v10, v00) = (g.potential(&x, p), h.v11(&x, p), h.v10(&x, p), h.v00(&x, p));
    let lhs = v11 / v;
    let rhs = 3. * (v10 / v00).powi(2);
    //Return left-hand-side / right-hand-side minus one
    (lhs.abs() - rhs.abs()).abs() / (lhs.abs() + rhs.abs())
  }

  #[inline(always)]
  pub fn consistency_only(x: [f64; 2], p: &[f64], h: &Hesse2D<'_>, g: &Potential<'_>) -> f64 {
    let (v, v11, v10, v00) = (g.potential(&x, p), h.v11(&x, p), h.v10(&x, p), h.v00(&x, p));
    let lhs = v11 / v - 3.;
    let rhs = 3. * (v00 / v10).powi(2) + (v00 / v) * (v10 / v00).powi(2);
    //Return left-hand-side / right-hand-side minus one
    (lhs.abs() - rhs.abs()).abs() / (lhs.abs() + rhs.abs())
  }

  #[inline(always)]
  pub fn flag_quantum_diff(x: [f64; 2], p: &[f64], accuracy: f64, g: &Potential<'_>) -> bool {
    let mut out = [0f64; 2];
    g.grad(&x, &p, &mut out);
    out.iter().all(|&x| x <= accuracy)
  }
}

#[pyfunction]
/// Evaluate the consistency condition ONLY, not considering any additional
/// parameters.
pub fn consistency_only(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize,
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate_lib(lib)?;
  validiate_p(lib, p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("{}Calculating consistency condition ONLY using {num_threads} threads.", *BADGE_INFO);
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out
    .as_slice_mut()
    .unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = ops::consistency_only;

  //(5b) setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out
      .iter_mut()
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g));
    } else {
      iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
    }
  } else {
    //Multi-threaded mode
    let threadpool =
      rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
    threadpool.install(move || {
      let iter = out
        .into_par_iter()
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g));
      } else {
        iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!("{}Calculation finished. Took {}.", *BADGE_INFO, indicatif::HumanDuration(start.elapsed()));

  Ok(())
}

#[pyfunction]
/// Evaluate the consistency condition ONLY using the rapid-turn approximations
/// from Anguelova & Lazaroiu's original paper (not considering any additional
/// parameters).
pub fn consistency_rapidturn_only(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize,
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate_lib(lib)?;
  validiate_p(lib, p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!(
    "{}Calculating consistency condition ONLY assuming rapid-turn using {num_threads} threads.",
    *BADGE_INFO
  );
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out
    .as_slice_mut()
    .unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = ops::consistency_rapidturn_only;

  //(5b) setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out
      .iter_mut()
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g));
    } else {
      iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
    }
  } else {
    //Multi-threaded mode
    let threadpool =
      rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
    threadpool.install(move || {
      let iter = out
        .into_par_iter()
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g));
      } else {
        iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!("{}Calculation finished. Took {}.", *BADGE_INFO, indicatif::HumanDuration(start.elapsed()));

  Ok(())
}

#[pyfunction]
/// Calculate the potential slow-roll parameter ε_V only
pub fn epsilon_v_only(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize,
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate_lib(lib)?;
  validiate_p(lib, p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!(
    "{}Calculating potential slow-roll parameter ε_V ONLY using {num_threads} threads.",
    *BADGE_INFO
  );
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out
    .as_slice_mut()
    .unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = ops::epsilon_v_only;

  //(5b) setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out
      .iter_mut()
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g));
    } else {
      iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
    }
  } else {
    //Multi-threaded mode
    let threadpool =
      rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
    threadpool.install(move || {
      let iter = out
        .into_par_iter()
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, &h, &g))
      } else {
        iter.for_each(|(x, val)| *val = op(x, p, &h, &g));
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!("{}Calculation finished. Took {}.", *BADGE_INFO, indicatif::HumanDuration(start.elapsed()));

  Ok(())
}

#[pyfunction]
/// Calculate everything calculable at once from the consistency condition. The
/// output array is 3D: the last axis consists of the following items (in order):
///   1. Consistency condition (lhs - rhs)
///   2. ε_V
///   3. ε_H
///   4. η_|| (eta parallel)
///   5. δ
///   6. ω
pub fn complete_analysis(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray3<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize,
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate_lib(lib)?;
  validiate_p(lib, p)?;
  if out.shape()[2] != 6 {
    Err(Error::Shape {
      expected: vec![out.shape()[0], out.shape()[1], 6],
      got: out.shape().to_vec(),
      msg: "Output array should be 3D. Last axis must have lenght 6".to_string(),
    })?
  }

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("{}Calculating full analysis using {num_threads} threads.", *BADGE_INFO);
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  // Fill output array
  let len = out.len() / 6;
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out
    .as_slice_mut()
    .unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  // shorthand for the actual calculation
  let op = ops::complete_analysis;

  // setup the threadpool (if necessary)
  if threads == 1 {
    //Single-threaded mode
    let iter = out
      .chunks_exact_mut(6)
      .enumerate()
      //(2a) convert flat index into array index
      .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
      //(2b) convert array index into field-space point
      .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
    if progress {
      iter.progress_with(set_pbar(len)).for_each(|(x, val)| op(x, val, &h, &g, p));
    } else {
      iter.for_each(|(x, val)| op(x, val, &h, &g, p));
    }
  } else {
    //Multi-threaded mode
    let threadpool =
      rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
    threadpool.install(move || {
      let iter = out
        .par_chunks_exact_mut(6)
        .enumerate()
        //(2a) convert flat index into array index
        .map(|(idx, val)| ([(idx / shape[1]) as f64, (idx % shape[1]) as f64], val))
        //(2b) convert array index into field-space point
        .map(move |(idx, val)| ([idx[0] * x_spacing + x_ofst, idx[1] * y_spacing + y_ofst], val));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| op(x, val, &h, &g, p));
      } else {
        iter.for_each(|(x, val)| op(x, val, &h, &g, p));
      }
    });
  }

  //(6) Report how long we took, and return.
  eprintln!("{}Calculation finished. Took {}.", *BADGE_INFO, indicatif::HumanDuration(start.elapsed()));

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
  accuracy: f64,
) -> PyResult<()> {
  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p
    .as_slice()
    .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
  let mut x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (_, g) = validate_lib(lib)?;
  validiate_p(lib, p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!(
    "{}Calculating zeros of the potential gradient using {} threads.",
    *BADGE_INFO,
    rayon::current_num_threads()
  );
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = x.len();
  let shape = &[x.shape()[0], x.shape()[1]];
  let iter = iter_array(x.as_slice_mut().unwrap(), &start_stop, shape);
  let op = ops::flag_quantum_diff;

  if progress {
    iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op(x, p, accuracy, &g));
  } else {
    iter.for_each(|(x, val)| *val = op(x, p, accuracy, &g));
  }

  //(6) Report how long we took, and return.
  eprintln!("{}Calculation finished. Took {}.", *BADGE_INFO, indicatif::HumanDuration(start.elapsed()));

  Ok(())
}

/// This module contains versions of the functions found in `crate::anguelova` that only evaluate
/// on specific points on the trajectory, rather than in a whole field-space plane. This has the
/// benifit of having complete control over at exactly which points the consistency condition will
/// be calculated. This is useful for evaluating the consistency condition on a already known
/// trajectory.
pub mod on_trajectory {
  use super::*;

  #[pyfunction]
  #[pyo3(name = "complete_analysis_on_trajectory")]
  /// Calculate everything calculable at once from the consistency condition. The 2D input array
  /// `x` should contain pairs of field-space coordinates (last axis should have length 2). For
  /// each field-space coordinate, the `out` array will be filled with:
  ///     1. Consistency condition (lhs - rhs)
  ///     2. ε_V
  ///     3. ε_H
  ///     4. η_
  ///     5. δ
  ///     6. ω
  /// In that order. Thus, the shape of the `x` array should always be (n,2) and the shape of the
  /// output array must always be (n,6). This function will return an error if this condition is
  /// not met.
  pub fn complete_analysis(
    lib: PyRef<crate::InflatoxPyDyLib>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray2<f64>,
    progress: bool,
    threads: usize,
  ) -> PyResult<()> {
    // get number of threads: 0 == rayon default
    let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

    // convert arguments to pure rust types
    let lib = &lib.0;
    let p = p
      .as_slice()
      .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
    let x = x.as_array();
    let mut out = out.as_array_mut();

    // validate that the input slices are all the correct shape etc.
    let (h, g) = validate_lib(lib)?;
    validiate_p(lib, p)?;
    if out.shape()[1] != 6 {
      Err(Error::Shape {
        expected: vec![out.shape()[0], 6],
        got: out.shape().to_vec(),
        msg: "Output array should be 2D. Last axis must have lenght 6".to_string(),
      })?
    }
    if out.shape()[0] != x.shape()[0] {
      Err(Error::Shape {
        expected: vec![x.shape()[0]],
        got: vec![out.shape()[0]],
        msg: "First axis of output array and field-space array should have the same length"
          .to_string(),
      })?
    }

    eprintln!("{}Calculating full analysis on trajectory using {num_threads} threads.", *BADGE_INFO);
    let _ = std::io::stderr().flush();
    let start = std::time::Instant::now();

    let op = ops::complete_analysis;

    let len = out.shape()[0];
    let out = out
      .as_slice_mut()
      .unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
    let x =
      x.as_slice().unwrap_or_else(|| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    if threads == 1 {
      //Single-threaded mode
      let iter = x.chunks_exact(2).zip(out.chunks_exact_mut(6));
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| op([x[0], x[1]], val, &h, &g, p));
      } else {
        iter.for_each(|(x, val)| op([x[0], x[1]], val, &h, &g, p));
      }
    } else {
      //Multi-threaded mode
      let threadpool =
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
      threadpool.install(move || {
        let iter = x.par_chunks_exact(2).zip(out.par_chunks_exact_mut(6));
        if progress {
          iter.progress_with(set_pbar(len)).for_each(|(x, val)| op([x[0], x[1]], val, &h, &g, p));
        } else {
          iter.for_each(|(x, val)| op([x[0], x[1]], val, &h, &g, p));
        }
      });
    }

    eprintln!(
      "{}Calculation finished. Took {}.",
      *BADGE_INFO,
      indicatif::HumanDuration(start.elapsed())
    );

    Ok(())
  }

  #[pyfunction]
  #[pyo3(name = "consistency_only_on_trajectory")]
  /// TODO: add docs
  pub fn consistency_only(
    lib: PyRef<crate::InflatoxPyDyLib>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray1<f64>,
    progress: bool,
    threads: usize,
  ) -> PyResult<()> {
    // get number of threads: 0 == rayon default
    let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

    // convert arguments to pure rust types
    let lib = &lib.0;
    let p = p
      .as_slice()
      .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
    let x = x.as_array();
    let out = out
      .as_slice_mut()
      .unwrap_or_else(|_| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    // validate that the input slices are all the correct shape etc.
    let (h, g) = validate_lib(lib)?;
    validiate_p(lib, p)?;
    if out.len() != x.shape()[0] {
      Err(Error::Shape {
        expected: vec![x.shape()[0]],
        got: vec![out.len()],
        msg: "Lenght of output array should equal the length of Axis(1) of the field-space array"
          .to_string(),
      })?
    }

    eprintln!(
      "{}Calculating consistency condition ONLY on trajectory using {num_threads} threads.",
      *BADGE_INFO
    );
    let _ = std::io::stderr().flush();
    let start = std::time::Instant::now();

    let op = ops::consistency_only;

    let len = out.len();
    let x = x
      .as_slice()
      .unwrap_or_else(|| panic!("{}FIELD-SPACE ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    if threads == 1 {
      //Single-threaded mode
      let iter = x.chunks_exact(2).zip(out.iter_mut());
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      } else {
        iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      }
    } else {
      //Multi-threaded mode
      let threadpool =
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
      threadpool.install(move || {
        let iter = x.par_chunks_exact(2).zip(out.par_iter_mut());
        if progress {
          iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        } else {
          iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        }
      });
    }

    eprintln!(
      "{}Calculation finished. Took {}.",
      *BADGE_INFO,
      indicatif::HumanDuration(start.elapsed())
    );

    Ok(())
  }

  #[pyfunction]
  #[pyo3(name = "consistency_rapidturn_only_on_trajectory")]
  /// TODO: add docs
  pub fn consistency_rapidturn_only(
    lib: PyRef<crate::InflatoxPyDyLib>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray1<f64>,
    progress: bool,
    threads: usize,
  ) -> PyResult<()> {
    // get number of threads: 0 == rayon default
    let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

    // convert arguments to pure rust types
    let lib = &lib.0;
    let p = p
      .as_slice()
      .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
    let x = x.as_array();
    let out = out
      .as_slice_mut()
      .unwrap_or_else(|_| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    // validate that the input slices are all the correct shape etc.
    let (h, g) = validate_lib(lib)?;
    validiate_p(lib, p)?;
    if out.len() != x.shape()[0] {
      Err(Error::Shape {
        expected: vec![x.shape()[0]],
        got: vec![out.len()],
        msg: "Lenght of output array should equal the length of Axis(1) of the field-space array"
          .to_string(),
      })?
    }

    eprintln!(
      "{}Calculating consistency condition (rapid turn approx.) ONLY on trajectory using {num_threads} threads.",
      *BADGE_INFO
    );
    let _ = std::io::stderr().flush();
    let start = std::time::Instant::now();

    let op = ops::consistency_rapidturn_only;

    let len = out.len();
    let x = x
      .as_slice()
      .unwrap_or_else(|| panic!("{}FIELD-SPACE ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    if threads == 1 {
      //Single-threaded mode
      let iter = x.chunks_exact(2).zip(out.iter_mut());
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      } else {
        iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      }
    } else {
      //Multi-threaded mode
      let threadpool =
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
      threadpool.install(move || {
        let iter = x.par_chunks_exact(2).zip(out.par_iter_mut());
        if progress {
          iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        } else {
          iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        }
      });
    }

    eprintln!(
      "{}Calculation finished. Took {}.",
      *BADGE_INFO,
      indicatif::HumanDuration(start.elapsed())
    );

    Ok(())
  }

  #[pyfunction]
  #[pyo3(name = "epsilon_v_only_on_trajectory")]
  /// TODO: add docs
  pub fn epsilon_v_only(
    lib: PyRef<crate::InflatoxPyDyLib>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    mut out: PyReadwriteArray1<f64>,
    progress: bool,
    threads: usize,
  ) -> PyResult<()> {
    // get number of threads: 0 == rayon default
    let num_threads = if threads != 0 { threads } else { rayon::current_num_threads() };

    // convert arguments to pure rust types
    let lib = &lib.0;
    let p = p
      .as_slice()
      .unwrap_or_else(|_| panic!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));
    let x = x.as_array();
    let out = out
      .as_slice_mut()
      .unwrap_or_else(|_| panic!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    // validate that the input slices are all the correct shape etc.
    let (h, g) = validate_lib(lib)?;
    validiate_p(lib, p)?;
    if out.len() != x.shape()[0] {
      Err(Error::Shape {
        expected: vec![x.shape()[0]],
        got: vec![out.len()],
        msg: "Lenght of output array should equal the length of Axis(1) of the field-space array"
          .to_string(),
      })?
    }

    eprintln!(
      "{}Calculating potential slow-roll parameter ε_V ONLY on trajectory using {num_threads} threads.",
      *BADGE_INFO
    );
    let _ = std::io::stderr().flush();
    let start = std::time::Instant::now();

    let op = ops::epsilon_v_only;

    let len = out.len();
    let x = x
      .as_slice()
      .unwrap_or_else(|| panic!("{}FIELD-SPACE ARRAY SHOULD BE C-CONTIGUOUS", *BADGE_PANIC));

    if threads == 1 {
      //Single-threaded mode
      let iter = x.chunks_exact(2).zip(out.iter_mut());
      if progress {
        iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      } else {
        iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
      }
    } else {
      //Multi-threaded mode
      let threadpool =
        rayon::ThreadPoolBuilder::new().num_threads(num_threads).build().map_err(Error::from)?;
      threadpool.install(move || {
        let iter = x.par_chunks_exact(2).zip(out.par_iter_mut());
        if progress {
          iter.progress_with(set_pbar(len)).for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        } else {
          iter.for_each(|(x, val)| *val = op([x[0], x[1]], p, &h, &g));
        }
      });
    }

    eprintln!(
      "{}Calculation finished. Took {}.",
      *BADGE_INFO,
      indicatif::HumanDuration(start.elapsed())
    );

    Ok(())
  }
}
