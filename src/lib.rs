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
