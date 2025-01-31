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

#![doc(
  html_logo_url = "https://raw.githubusercontent.com/smups/inflatox/dev/logos/logo.png?raw=true"
)]

mod anguelova;
mod background_solver;
mod dylib;
mod err;
mod hesse_bindings;
mod inflatox_version;
// mod rk;

use std::sync::LazyLock;

use dylib::InflatoxDylib;
use hesse_bindings::*;
use inflatox_version::InflatoxVersion;

use ndarray as nd;
use numpy::{
  PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
};
use pyo3::prelude::*;

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

/// Version of Inflatox ABI that this crate is compatible with
pub static V_INFLX_ABI: InflatoxVersion = InflatoxVersion::new([5, 0, 0]);

// Badge in front of inflatox output
pub static BADGE_INFO: LazyLock<console::StyledObject<&'static str>> = LazyLock::new(|| {
  let magenta = console::Style::new().magenta().bold();
  magenta.apply_to("[Inflatox Info]\n")
});

pub static BADGE_WARN: LazyLock<console::StyledObject<&'static str>> = LazyLock::new(|| {
  let magenta = console::Style::new().yellow().bold();
  magenta.apply_to("[Inflatox Warning]\n")
});

pub static BADGE_PANIC: LazyLock<console::StyledObject<&'static str>> = LazyLock::new(|| {
  let red = console::Style::new().red().bold();
  red.apply_to("[Inflatox PANIC]\n")
});

#[pymodule]
/// PyO3 wrapper for libinflx_rs rust api
fn libinflx_rs(_py: Python<'_>, pymod: &Bound<PyModule>) -> PyResult<()> {
  use anguelova::*;
  pymod.add_class::<InflatoxPyDyLib>()?;
  pymod.add_function(wrap_pyfunction!(open_inflx_dylib, pymod)?)?;

  pymod.add_function(wrap_pyfunction!(flag_quantum_dif_py, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(consistency_only, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(consistency_rapidturn_only, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(epsilon_v_only, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(complete_analysis, pymod)?)?;

  pymod.add_function(wrap_pyfunction!(on_trajectory::complete_analysis, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(on_trajectory::consistency_only, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(on_trajectory::consistency_rapidturn_only, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(on_trajectory::epsilon_v_only, pymod)?)?;

  pymod.add_function(wrap_pyfunction!(background_solver::solve_eom_rk4, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(background_solver::solve_eom_rkf, pymod)?)?;

  Ok(())
}

#[pyclass]
/// Python wrapper for `InflatoxDyLib`
pub struct InflatoxPyDyLib(pub InflatoxDylib);

#[pyfunction]
fn open_inflx_dylib(lib_path: &str, check_basis: bool) -> PyResult<InflatoxPyDyLib> {
  let dylib = InflatoxPyDyLib(InflatoxDylib::open(lib_path)?);
  if check_basis {
    dylib.validate_basis_at_random()?;
  };
  Ok(dylib)
}

pub fn convert_start_stop(
  start_stop: nd::ArrayView2<f64>,
  n_fields: usize,
) -> Result<Vec<[f64; 2]>> {
  if start_stop.shape().len() != 2
    || start_stop.shape()[1] != n_fields
    || start_stop.shape()[0] != 2
  {
    Err(Error::Shape {
      expected: vec![2, n_fields],
      got: start_stop.shape().to_vec(),
      msg: "start_stop array should have 2 rows and as many columns as there are fields"
        .to_string(),
    })
  } else {
    Ok(
      start_stop
        .axis_iter(nd::Axis(0))
        .map(|start_stop| [start_stop[0], start_stop[1]])
        .collect::<Vec<_>>(),
    )
  }
}

impl InflatoxPyDyLib {
  fn validate_basis_at_random(&self) -> Result<()> {
    let accuracy = 1e-3;
    let p = {
      let mut vec = vec![0f64; self.0.n_pars()];
      vec.iter_mut().for_each(|elem| *elem = 10. * (-1. + 2. * rand::random::<f64>()));
      vec
    };
    let mut vi = vec![0f64; self.0.n_fields()];
    let mut vj = vec![0f64; self.0.n_fields()];
    let inner_prod = self.0.get_inner_prod_fn()?;
    let basis_fns = self.0.get_basis_fns()?;

    let mut failed = 0;
    let num_points = 100;
    for _ in 0..num_points {
      let x = {
        let mut vec = vec![0f64; self.0.n_fields()];
        vec.iter_mut().for_each(|elem| *elem = -1. + 2. * rand::random::<f64>());
        vec
      };

      let mut encountered_nan = false;
      for i in 0..self.0.n_fields() {
        for j in i..self.0.n_fields() {
          // compute v1 and v2
          unsafe { (basis_fns[i])(x.as_ptr(), p.as_ptr(), vi.as_mut_ptr()) };
          unsafe { (basis_fns[j])(x.as_ptr(), p.as_ptr(), vj.as_mut_ptr()) };
          let inner_prod = unsafe { inner_prod(x.as_ptr(), p.as_ptr(), vi.as_ptr(), vj.as_ptr()) };

          if i == j {
            if !inner_prod.is_normal() {
              eprintln!(
              "{}Norm of basisvector {i} is {inner_prod} at field-space point {x:.03?}. v{i}={vi:.03?}\nAre we outside the model's domain?",
              *BADGE_WARN
            );
              encountered_nan = true;
            } else if (inner_prod - 1.).abs() >= accuracy {
              return Err(Error::BasisNorm { norm: inner_prod, vector: i as _, point: x });
            }
          } else {
            if !inner_prod.is_normal() && inner_prod != 0.0 {
              eprintln!("{}w{i}•w{j} = {inner_prod} at field-space point {x:.03?}.\nv{i}={vi:.03?}\nv{j}={vj:.03?}\nAre we outside the model's domain?", *BADGE_WARN);
              encountered_nan = true;
            } else if inner_prod.abs() >= accuracy {
              return Err(Error::BasisOth { inner_prod, vectors: (i as _, j as _), point: x });
            }
          }
        }
      }

      if encountered_nan {
        failed += 1;
      }
    }

    if failed != 0 {
      eprintln!("{}Inflatox was unable to verify basis orthonormality at {failed} out of {num_points} tested points.\nThis could be indicative of a defective model.\nUsed parameter values: p={p:.03?}", *BADGE_WARN);
    }

    Ok(())
  }
}

#[pymethods]
impl InflatoxPyDyLib {
  fn validate_basis_on_domain(
    &self,
    num_points: PyReadonlyArray1<u32>,
    p: PyReadonlyArray1<f64>,
    start_stop: PyReadonlyArray2<f64>,
    accuracy: f64,
  ) -> Result<()> {
    // Convert the PyArrays to nd::Arrays
    eprintln!(
      "{}Validating basis orthonormality on specified domain. This may take a while...",
      *BADGE_INFO
    );
    let p = p.as_array();
    let num_points = num_points.as_array();
    let start_stop = start_stop.as_array();

    // Make sure that the number of supplied fields matches the number
    // specified by the dynamic lib
    if num_points.len() != self.0.n_fields() {
      return Err(Error::Shape {
        expected: Vec::new(),
        got: num_points.shape().to_vec(),
        msg:
          "expected an array with with the same number of axes as there are field-space coordinates"
            .to_string(),
      });
    }

    // Convert start_stop array
    let start_stop = convert_start_stop(start_stop.view(), self.0.n_fields())?;

    // Make sure that the number of supplied model parameters matches the number
    // specified by the dynamic lib
    if p.shape() != [self.0.n_pars()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_pars()],
        got: p.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are model parameters".to_string(),
      });
    }
    let p = p.as_slice().unwrap();

    let mut vi = vec![0f64; self.0.n_fields()];
    let mut vj = vec![0f64; self.0.n_fields()];
    let inner_prod = self.0.get_inner_prod_fn()?;
    let basis_fns = self.0.get_basis_fns()?;

    let point = start_stop.iter().map(|start_stop| start_stop[0]).collect::<Vec<_>>();
    let mut failed = 0;
    for (axis_idx, &axis_len) in num_points.into_iter().enumerate() {
      let [start, stop] = start_stop[axis_idx];
      let spacing = (stop - start) / axis_len as f64;
      (0..axis_len).into_iter().map(|idx| {
        let mut point = point.clone();
        point[axis_idx] = stop + spacing * idx as f64;
        point
      }).try_for_each(|point| {
      let mut encountered_nan = false;
      for i in 0..self.0.n_fields() {
        for j in i..self.0.n_fields() {
          // compute v1 and v2
          unsafe { (basis_fns[i])(point.as_ptr(), p.as_ptr(), vi.as_mut_ptr()) };
          unsafe { (basis_fns[j])(point.as_ptr(), p.as_ptr(), vj.as_mut_ptr()) };
          let inner_prod = unsafe { inner_prod(point.as_ptr(), p.as_ptr(), vi.as_ptr(), vj.as_ptr()) };
          if i == j {
            if !inner_prod.is_normal() {
              eprintln!(
                "{}Norm of basisvector {i} is {inner_prod} at field-space point {point:.03?}. v{i}={vi:.03?}\nAre we outside the model's domain?",
                *BADGE_WARN
              );
              encountered_nan = true;
            } else if (inner_prod - 1.).abs() >= accuracy {
              return Err(Error::BasisNorm { norm: inner_prod, vector: i as _, point });
            }
          } else {
            if !inner_prod.is_normal() && inner_prod != 0.0 {
              eprintln!("{}w{i}•w{j} = {inner_prod} at field-space point {point:.03?}.\nv{i}={vi:.03?}\nv{j}={vj:.03?}\nAre we outside the model's domain?", *BADGE_WARN);
              encountered_nan = true;
            } else if inner_prod.abs() >= accuracy {
              return Err(Error::BasisOth { inner_prod, vectors: (i as _, j as _), point });
            }
          }
        }
      }

      if encountered_nan {
        failed += 1;
      }

      Ok(())})?;

      if failed != 0 {
        eprintln!("{}Inflatox was unable to verify basis orthonormality at {failed} out of {} tested points.\nThis could be indicative of a defective model.\nUsed parameter values: p={p:.03?}",
          *BADGE_WARN,
          num_points.iter().fold(1., |acc, &num_points| acc * num_points as f64)
        );
      }
    }

    Ok(())
  }

  fn potential(&self, x: PyReadonlyArrayDyn<f64>, p: PyReadonlyArrayDyn<f64>) -> Result<f64> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array();

    //(3) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape() != [self.0.n_fields()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_fields()],
        got: x.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are field-space coordinates"
          .to_string(),
      });
    }
    let x = x.as_slice().unwrap();

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != [self.0.n_pars()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_pars()],
        got: p.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are model parameters".to_string(),
      });
    }
    let p = p.as_slice().unwrap();

    //(4) Calculate
    Ok(Potential::new(&self.0)?.potential(x, p))
  }

  fn potential_array(
    &self,
    mut x: PyReadwriteArrayDyn<f64>,
    p: PyReadonlyArrayDyn<f64>,
    start_stop: PyReadonlyArray2<f64>,
  ) -> Result<()> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array_mut();
    let start_stop = start_stop.as_array();

    //(1) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape().len() != self.0.n_fields() {
      return Err(Error::Shape {
        expected: Vec::new(),
        got: x.shape().to_vec(),
        msg:
          "expected an array with with the same number of axes as there are field-space coordinates"
            .to_string(),
      });
    }

    //(2) Convert start_stop array
    let start_stop = convert_start_stop(start_stop.view(), self.0.n_fields())?;

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != [self.0.n_pars()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_pars()],
        got: p.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are model parameters".to_string(),
      });
    }
    let p = p.as_slice().unwrap();

    //(4) Evaluate the potential
    Potential::new(&self.0)?.potential_array(x, p, &start_stop);

    Ok(())
  }

  fn hesse<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    p: PyReadonlyArrayDyn<f64>,
  ) -> Result<Bound<'py, PyArray2<f64>>> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array();

    //(3) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape() != [self.0.n_fields()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_fields()],
        got: x.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are field-space coordinates"
          .to_string(),
      });
    }
    let x = x.as_slice().unwrap();

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != [self.0.n_pars()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_pars()],
        got: p.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are model parameters".to_string(),
      });
    }
    let p = p.as_slice().unwrap();

    //(4) Calculate
    Ok(PyArray2::from_owned_array(py, Hesse::new(&self.0)?.hesse(x, p)))
  }

  fn hesse_array<'py>(
    &self,
    py: Python<'py>,
    nx: PyReadonlyArray1<usize>,
    p: PyReadonlyArrayDyn<f64>,
    start_stop: PyReadonlyArray2<f64>,
  ) -> Result<Bound<'py, PyArrayDyn<f64>>> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let nx = nx.as_array();
    let nx = nx.as_slice().unwrap();
    let start_stop = start_stop.as_array();

    //(1) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if nx.len() != self.0.n_fields() {
      return Err(Error::Shape {
        expected: vec![self.0.n_fields()],
        got: vec![nx.len()],
        msg: "expected a 1D array with as many elements as there are field-space coordinates"
          .to_string(),
      });
    }

    //(2) Convert start_stop array
    let start_stop = convert_start_stop(start_stop.view(), self.0.n_fields())?;

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != [self.0.n_pars()] {
      return Err(Error::Shape {
        expected: vec![self.0.n_pars()],
        got: p.shape().to_vec(),
        msg: "expected a 1D array with as many elements as there are model parameters".to_string(),
      });
    }
    let p = p.as_slice().unwrap();

    //(4) Evaluate the hesse matrix
    let out = Hesse::new(&self.0)?.hesse_array(nx, p, &start_stop);
    Ok(PyArrayDyn::from_owned_array(py, out))
  }
}
