mod anguelova;
mod hesse_bindings;
mod inflatox_version;

use hesse_bindings::{open_inflx_dylib, InflatoxPyDyLib};
use inflatox_version::InflatoxVersion;

use pyo3::{create_exception, exceptions::PyException, prelude::*};

pub(crate) const V_INFLX: InflatoxVersion = InflatoxVersion::new([0, 1, 0]);

//Register errors
create_exception!(libinflx_rs, ShapeError, PyException);

#[cfg(feature = "pyo3_extension_module")]
#[pymodule]
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
fn libinflx_rs(py: Python<'_>, pymod: &PyModule) -> PyResult<()> {
  pymod.add_class::<InflatoxPyDyLib>()?;
  pymod.add_function(wrap_pyfunction!(open_inflx_dylib, pymod)?)?;
  pymod.add_function(wrap_pyfunction!(anguelova::anguelova, pymod)?)?;

  //Register exceptions
  pymod.add("DimensionalityError", py.get_type::<ShapeError>())?;
  Ok(())
}
