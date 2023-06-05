
use ndarray as nd;
use numpy::{PyReadwriteArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hesse_bindings::{InflatoxPyDyLib, HesseNd, Hesse2D, raise_shape_err, convert_start_stop};

#[pyfunction]
pub(crate) fn anguelova(
  lib: PyRef<InflatoxPyDyLib>,
  p: PyReadonlyArray1<f64>,
  mut x: PyReadwriteArray2<f64>,
  start_stop: PyReadonlyArray2<f64>,
) -> PyResult<()> {
  //(0) Convert the PyArrays to nd::Arrays
  let h = HesseNd::new(&lib.0);
  let p = p.as_array();
  let x = x.as_array_mut();
  let start_stop = start_stop.as_array();

  //(1) Make sure we have a two field model
  if !h.get_n_fields() == 2 {
    raise_shape_err(format!("the Anguelova consistency condition requires a 2-field model. Received a {}-field model.", h.get_n_fields()))?;
  }
  let h = Hesse2D::new(h);

  //(2) Make sure the field-space array is actually 2d
  if x.shape().len() != 2 {
    raise_shape_err(format!("expected a 2D field-space array. Found array with shape {:?}", x.shape()))?;
  }

  //(3) Make sure that the number of supplied model parameters matches the number
  //specified by the dynamic lib
  if p.shape() != &[h.get_n_params()] {
    raise_shape_err(format!("model expected {} parameters, got {}", h.get_n_params(), p.shape().len()))?;
  }
  let p = p.as_slice().unwrap();

  //(4) Convert start-stop
  let start_stop = convert_start_stop(start_stop, 2)?;
  
  //(5) evaluate anguelova's condition
  anguelova_raw(h, x, p, &start_stop);

  Ok(())
}

pub(crate) fn anguelova_raw(
  h: Hesse2D,
  x: nd::ArrayViewMut2<f64>,
  p: &[f64],
  start_stop: &[[f64; 2]]
) {
  //(1) Convert start-stop ranges
  let x_spacing = (start_stop[0][1] - start_stop[0][0]) / x.shape()[0] as f64;
  let x_ofst = start_stop[0][0] as f64 * x_spacing;
  let y_spacing = (start_stop[1][1] - start_stop[1][0]) / x.shape()[1] as f64;
  let y_ofst = start_stop[1][0] as f64 * y_spacing;

  //(2) Fill output array
  nd::Zip::indexed(x)
    .into_par_iter()
    //(2a) Convert indices to field-space coordinates
    .map(|(idx, val)| (
      [idx.0 as f64 * x_spacing + x_ofst, idx.1 as f64 * y_spacing + y_ofst],
      val
    ))
    //(2b) evaluate consistency condition at every field-space point
    .for_each(|(ref x, val)| *val = {
      let lhs = 3.0 * h.potential(x, p) * (h.v00(x, p) / h.v01(x, p).powi(2)).powi(2);
      let rhs = h.v11(x, p);
      lhs - rhs
    });
}