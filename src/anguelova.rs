use ndarray as nd;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hesse_bindings::{
    convert_start_stop, raise_shape_err, Hesse2D, HesseNd, InflatoxDylib, InflatoxPyDyLib,
};

#[cfg(feature = "pyo3_extension_module")]
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
        raise_shape_err(format!(
            "expected a 2D field-space array. Found array with shape {:?}",
            x.shape()
        ))?;
    }

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != &[h.get_n_params()] {
        raise_shape_err(format!(
            "model expected {} parameters, got {}",
            h.get_n_params(),
            p.shape().len()
        ))?;
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
    start_stop: &[[f64; 2]],
) {
    //(1) Convert start-stop ranges
    let (x_spacing, x_ofst) = {
        let x_start = start_stop[0][0];
        let x_stop = start_stop[0][1];
        let x_spacing = (x_stop - x_start) / x.shape()[0] as f64;
        (x_spacing, x_start)
    };
    let (y_spacing, y_ofst) = {
        let y_start = start_stop[1][0];
        let y_stop = start_stop[1][1];
        let y_spacing = (y_stop - y_start) / x.shape()[1] as f64;
        (y_spacing, y_start)
    };

    //(2) Fill output array
    nd::Zip::indexed(x)
        .into_par_iter()
        //(2a) Convert indices to field-space coordinates
        .map(|(idx, val)| {
            (
                [
                    idx.0 as f64 * x_spacing + x_ofst,
                    idx.1 as f64 * y_spacing + y_ofst,
                ],
                val,
            )
        })
        //(2b) evaluate consistency condition at every field-space point
        .for_each(|(ref x, val)| {
            *val = {
                let lhs = 3.0 * (h.v00(x, p) / h.v01(x, p).powi(2)).powi(2);
                let rhs = h.v11(x, p) / h.potential(x, p);
                lhs - rhs
            }
        });
}

#[test]
fn anguelova_performance() {
    let n = 10_000;
    let mut out = nd::Array2::zeros((n, n));
    let start_stop = [[-1000.0, 1000.0], [-1000.0, 1000.0]];
    let lib = InflatoxDylib::open("/tmp/libinflx_autoc_z1lc1jur.so").unwrap();
    let h = Hesse2D::new(HesseNd::new(&lib));
    let p = &[12.0, 3.0, 4.0, -12.0];
    anguelova_raw(h, out.view_mut(), p, &start_stop);
    println!("{out:?}");
}
