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

use pyo3::exceptions::PyException;

use crate::inflatox_version::InflatoxVersion;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibInflxRsErr {
  IoErr { lib_path: String, msg: String },
  MissingSymbolErr { symbol: Vec<u8>, lib_path: String },
  VersionErr(InflatoxVersion),
  RayonErr(String),
  ShapeErr { expected: Vec<usize>, got: Vec<usize>, msg: String },
}

impl std::fmt::Display for LibInflxRsErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use LibInflxRsErr::*;
    #[cfg_attr(rustfmt, rustfmt_skip)]
    match self {
      IoErr { lib_path, msg } => write!(f, "Could not load Inflatox Compilation Artefact (path: {lib_path}). Error: \"{msg}\""),
      MissingSymbolErr { symbol, lib_path } => {
        if let Ok(string) = std::str::from_utf8(&symbol) {
          write!(f, "Could not find symbol \"{string}\" in {lib_path}")
        } else {
          write!(f, "Could not find symbol {symbol:?} in {lib_path}")
        }
      },
      VersionErr(v) => write!(f, "Cannot load Inflatox Compilation Artefact compiled for Inflatox ABI {v} using current Inflatox installation ({})", crate::V_INFLX_ABI),
      RayonErr(msg) => write!(f, "Could not initialise threadpool. Error: \"{msg}\""),
      ShapeErr { expected, got, msg } => write!(f, "Expected array with shape {expected:?}, received array with shape {got:?}. Context: {msg}")
    }
  }
}
impl std::error::Error for LibInflxRsErr {}

impl From<LibInflxRsErr> for pyo3::PyErr {
  fn from(err: LibInflxRsErr) -> Self {
    use pyo3::exceptions::{PyIOError, PySystemError};
    use LibInflxRsErr::*;
    let msg = format!("{err}");
    match err {
      IoErr { .. } => PyIOError::new_err(msg),
      MissingSymbolErr { .. } | VersionErr(_) | RayonErr(_) => PySystemError::new_err(msg),
      ShapeErr { .. } => PyException::new_err(msg),
    }
  }
}

impl From<rayon::ThreadPoolBuildError> for LibInflxRsErr {
  fn from(err: rayon::ThreadPoolBuildError) -> Self {
    Self::RayonErr(format!("{err}"))
  }
}
