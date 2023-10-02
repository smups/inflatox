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

#![doc(
  html_logo_url = "https://raw.githubusercontent.com/smups/inflatox/dev/logos/logo.png?raw=true"
)]

mod anguelova;
mod err;
mod hesse_bindings;
mod inflatox_version;

use hesse_bindings::InflatoxDylib;
use inflatox_version::InflatoxVersion;

use ndarray as nd;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::{create_exception, exceptions::PyException, prelude::*};

/// Version of Inflatox ABI that this crate is compatible with
pub const V_INFLX_ABI: InflatoxVersion = InflatoxVersion::new([2, 0, 0]);

//Register errors
create_exception!(libinflx_rs, ShapeError, PyException);

#[pymodule]
/// PyO3 wrapper for libinflx_rs rust api
fn libinflx_rs(py: Python<'_>, pymod: &PyModule) -> PyResult<()> {
  pymod.add_class::<InflatoxPyDyLib>()?;
  pymod.add_function(wrap_pyfunction!(open_inflx_dylib, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(anguelova::anguelova_py, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(anguelova::delta_py, pymod)?)?;

  //Register exceptions
  pymod.add("DimensionalityError", py.get_type::<ShapeError>())?;
  Ok(())
}

#[pyclass]
/// Python wrapper for `InflatoxDyLib`
pub struct InflatoxPyDyLib(pub InflatoxDylib);

#[pyfunction]
fn open_inflx_dylib(lib_path: &str) -> PyResult<InflatoxPyDyLib> {
  Ok(InflatoxPyDyLib(InflatoxDylib::open(lib_path)?))
}

/// Utility function to easily raise a shape error.
pub fn raise_shape_err<T>(err: String) -> PyResult<T> {
  Err(ShapeError::new_err(err))
}

pub fn convert_start_stop(
  start_stop: nd::ArrayView2<f64>,
  n_fields: usize,
) -> PyResult<Vec<[f64; 2]>> {
  if start_stop.shape().len() != 2
    || start_stop.shape()[1] != n_fields
    || start_stop.shape()[0] != 2
  {
    raise_shape_err(format!("start_stop array should have 2 rows and as many columns as there are fields ({}). Got start_stop with shape {:?}", n_fields, start_stop.shape()))?;
  }
  let start_stop = start_stop
    .axis_iter(nd::Axis(0))
    .map(|start_stop| [start_stop[0], start_stop[1]])
    .collect::<Vec<_>>();
  Ok(start_stop)
}

#[pymethods]
impl InflatoxPyDyLib {
  fn potential(&self, x: PyReadonlyArrayDyn<f64>, p: PyReadonlyArrayDyn<f64>) -> PyResult<f64> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array();

    //(3) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape() != &[self.0.get_n_fields() as usize] {
      raise_shape_err(format!("expected a 1D array with {} elements as field-space coordinates. Found array with shape {:?}", self.0.get_n_fields(), x.shape()))?;
    }
    let x = x.as_slice().unwrap();

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != &[self.0.get_n_params() as usize] {
      raise_shape_err(format!(
        "expected a 1D array with {} elements as parameters set. Found array with shape {:?}",
        self.0.get_n_params(),
        p.shape()
      ))?;
    }
    let p = p.as_slice().unwrap();

    //(4) Calculate
    Ok(self.0.potential(x, p))
  }

  fn potential_array(
    &self,
    mut x: PyReadwriteArrayDyn<f64>,
    p: PyReadonlyArrayDyn<f64>,
    start_stop: PyReadonlyArray2<f64>,
  ) -> PyResult<()> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array_mut();
    let start_stop = start_stop.as_array();

    //(1) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape().len() != self.0.get_n_fields() as usize {
      raise_shape_err(format!(
        "expected an array with {} axes as field-space coordinates. Found array with shape {:?}",
        self.0.get_n_fields(),
        x.shape()
      ))?;
    }

    //(2) Convert start_stop array
    let start_stop = convert_start_stop(start_stop.view(), self.0.get_n_fields())?;

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != &[self.0.get_n_params() as usize] {
      raise_shape_err(format!(
        "expected a 1D array with {} elements as parameters set. Found array with shape {:?}",
        self.0.get_n_params(),
        p.shape()
      ))?;
    }
    let p = p.as_slice().unwrap();

    //(4) Evaluate the potential
    self.0.potential_array(x, p, &start_stop);

    Ok(())
  }

  fn hesse<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    p: PyReadonlyArrayDyn<f64>,
  ) -> PyResult<&'py PyArray2<f64>> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array();

    //(3) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape() != &[self.0.get_n_fields() as usize] {
      raise_shape_err(format!("expected a 1D array with {} elements as field-space coordinates. Found array with shape {:?}", self.0.get_n_fields(), x.shape()))?;
    }
    let x = x.as_slice().unwrap();

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != &[self.0.get_n_params() as usize] {
      raise_shape_err(format!(
        "expected a 1D array with {} elements as parameters set. Found array with shape {:?}",
        self.0.get_n_params(),
        p.shape()
      ))?;
    }
    let p = p.as_slice().unwrap();

    //(4) Calculate
    Ok(PyArray2::from_owned_array(py, self.0.hesse(x, p)))
  }
}
