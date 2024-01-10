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
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyReadwriteArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hesse_bindings::{Hesse2D, InflatoxDylib, Grad};
use crate::PANIC_BADGE;

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
pub fn consistency_only(
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
  let p = p.as_slice().expect(&format!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
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
  let out = out.as_slice_mut().expect(&format!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    let (v, v11, v10, v00) = (h.potential(x, p), h.v11(x, p), h.v10(x, p), h.v00(x, p));
    let lhs = v11/v - 3.;
    let rhs = 3.*(v00/v10).powi(2) + (v00/v)*(v10/v00).powi(2);
    //Return left-hand-side / right-hand-side minus one
    *val = ((lhs/rhs).abs() - 1.).abs()
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
pub fn consistency_rapidturn_only(
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
  let p = p.as_slice().expect(&format!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
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
  let out = out.as_slice_mut().expect(&format!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    let (v, v11, v10, v00) = (h.potential(x, p), h.v11(x, p), h.v10(x, p), h.v00(x, p));
    let lhs = v11/v;
    let rhs = 3. * (v10/v00).powi(2);
    //Return left-hand-side / right-hand-side minus one
    *val = ((lhs/rhs).abs() - 1.).abs()
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
/// Calculate the potential slow-roll parameter ε_V only
pub fn epsilon_v_only(
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
  let p = p.as_slice().expect(&format!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate(lib, out.view(), p)?;

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating potential slow-roll parameter ε_V ONLY using {num_threads} threads.");
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out.as_slice_mut().expect(&format!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  let op = |(ref x, val): ([f64; 2], &mut f64)| {
    *val = g.grad_square(x, p) / h.potential(x, p).powi(2);
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
/// Calculate everything calculable at once from the consistency condition. The
/// output array is 3D: the last axis consists of the following items (in order):
///   1. Consistency condition (lhs - rhs)
///   2. ε_V
///   3. ε_H
///   4. η_H
///   5. δ
///   6. ω
pub fn complete_analysis(
  lib: PyRef<crate::InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut out: PyReadwriteArray3<f64>,
  start_stop: PyReadonlyArray2<f64>,
  progress: bool,
  threads: usize
) -> PyResult<()> {
  //(0) Set number of threads to use
  let num_threads = if threads != 0 { threads } else {rayon::current_num_threads()};

  //(1) Convert the PyArrays to nd::Arrays
  let lib = &lib.0;
  let p = p.as_slice().expect(&format!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let mut out = out.as_array_mut();
  let start_stop = start_stop.as_array();

  //(2) Validate that the input is usable for evaluating Anguelova-Lazaroiu's condition
  let (h, g) = validate(lib, out.slice(nd::s![..,..,0]), p)?;
  if out.shape()[2] != 6 {
    crate::raise_shape_err(format!(
      "expected output array with shape (x,y,6). Found array with shape {:?}",
      out.shape()
    ))?;
  }

  //(3) Convert start-stop
  let start_stop = crate::convert_start_stop(start_stop, 2)?;

  //(4) Say hello
  eprintln!("[Inflatox] Calculating full analysis using {num_threads} threads.");
  let _ = std::io::stderr().flush();
  let start = std::time::Instant::now();

  //(5) Fill output array
  let len = out.len();
  let shape = &[out.shape()[0], out.shape()[1]];
  let out = out.as_slice_mut().expect(&format!("{}OUTPUT ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
  let (x_spacing, y_spacing, x_ofst, y_ofst) = convert_ranges(&start_stop, shape);

  //(5a) Define the calculation
  fn op(x: [f64; 2], val: &mut [f64], h: &Hesse2D<'_>, g: &Grad<'_>, p: &[f64]) {
    let (v, v11, v10, v00) = (h.potential(&x, p), h.v11(&x, p), h.v10(&x, p), h.v00(&x, p));
    //(1) Calculate consistency condition
    let consistency = {
      let lhs = v11/v;
      let rhs = 3. + 3.*(v00/v10).powi(2) + (v00/v)*(v10/v00).powi(2);
      ((lhs/rhs).abs() - 1.).abs()
    };
    //(2) Calculate ε_V
    let epsilon_v = g.grad_square(&x, p) / v.powi(2);
    //(3) Calculate Vtt
    let vtt = (v00*v10.powi(2) + v11*v00.powi(2) - 2.*v00*v10.powi(2)) / (v00.powi(2) + v10.powi(2));
    //(4) Calculate ε_H
    let epsilon_h = (3. * epsilon_v) / (epsilon_v + 3. + vtt/(3.*v));
    //(5) Calculate η_H
    let eta_h = (3.*(3.-epsilon_h)).sqrt() - 3.;
    //(6) Calculate δ
    let delta = v10.atan2(v00);
    //(7) Calculate ω
    let omega = ((vtt/v) * (3.-epsilon_h)).sqrt();

    val.swap_with_slice(&mut [consistency, epsilon_v, epsilon_h, eta_h, delta, omega]);
  }

  //(5b) setup the threadpool (if necessary)
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
    let threadpool = rayon::ThreadPoolBuilder::new()
      .num_threads(num_threads)
      .build()
      .expect("[LIBINFLX_RS_PANIC]: COULD NOT INITIALISE THREADPOOL");
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
  let p = p.as_slice().expect(&format!("{}PARAMETER ARRAY SHOULD BE C-CONTIGUOUS", *PANIC_BADGE));
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
