use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::hesse_bindings::{InflatoxPyDyLib, HesseNd, Hesse2D, raise_shape_err, convert_start_stop};

#[pyfunction]
pub(crate) fn anguelova(
  lib: PyRef<InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  start_stop: PyReadonlyArray2<f64>,
) -> PyResult<PyArray2<f64>> {
  //(0) Convert the PyArrays to nd::Arrays
  let h = HesseNd::new(&lib.0);
  let p = p.as_array();
  let start_stop = start_stop.as_array();

  //(1) Make sure the dimensionalities all work out
  if !h.get_n_fields() == 2 {
    raise_shape_err(format!("the Anguelova consistency requires a 2-field model. Received a {}-field model.", h.get_n_fields()))?;
  }
  let h = Hesse2D::new(h);

  if p.shape() != &[h.get_n_params()] {
    raise_shape_err(format!("model expected {} parameters, got {}", h.get_n_params(), p.shape().len()))?;
  }
  let p = p.as_standard_layout().as_slice().unwrap();
  let start_stop = convert_start_stop(start_stop, 2)?;
  
  todo!()
}