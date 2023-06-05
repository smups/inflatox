mod hesse_bindings;
mod inflatox_version;
mod anguelova;

use hesse_bindings::{InflatoxPyDyLib, open_inflx_dylib};
use inflatox_version::InflatoxVersion;

use pyo3::{prelude::*, create_exception, exceptions::PyException};

pub(crate) const V_INFLX: InflatoxVersion = InflatoxVersion::new([0,1,0]);

//Register errors
create_exception!(libinflx_rs, ShapeError, PyException);

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn libinflx_rs(py: Python<'_>, pymod: &PyModule) -> PyResult<()> {
  pymod.add_function(wrap_pyfunction!(sum_as_string, pymod)?)?;
  pymod.add_class::<InflatoxPyDyLib>()?;
  pymod.add_function(wrap_pyfunction!(open_inflx_dylib, pymod)?)?;

  //Register exceptions
  pymod.add("DimensionalityError", py.get_type::<ShapeError>())?;
  Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
  Ok((a + b).to_string())
}