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

use std::ffi::{c_char, c_int};

use pyo3::exceptions::PyException;

use crate::inflatox_version::InflatoxVersion;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibInflxRsErr {
  Io { lib_path: String, msg: String },
  MissingSymbol { symbol: Vec<u8>, lib_path: String },
  Version(InflatoxVersion),
  Rayon(String),
  Shape { expected: Vec<usize>, got: Vec<usize>, msg: String },
}

impl std::fmt::Display for LibInflxRsErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use LibInflxRsErr::*;
    match self {
      Io { lib_path, msg } => write!(f, "Could not load Inflatox Compilation Artefact (path: {lib_path}). Error: \"{msg}\""),
      MissingSymbol { symbol, lib_path } => {
        if let Ok(string) = std::str::from_utf8(symbol) {
          write!(f, "Could not find symbol \"{string}\" in {lib_path}")
        } else {
          write!(f, "Could not find symbol {symbol:?} in {lib_path}")
        }
      },
      Version(v) => write!(f, "Cannot load Inflatox Compilation Artefact compiled for Inflatox ABI {v} using current Inflatox installation ({})", crate::V_INFLX_ABI),
      Rayon(msg) => write!(f, "Could not initialise threadpool. Error: \"{msg}\""),
      Shape { expected, got, msg } => write!(f, "Expected array with shape {expected:?}, received array with shape {got:?}. Context: {msg}")
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
      Io { .. } => PyIOError::new_err(msg),
      MissingSymbol { .. } | Version(_) | Rayon(_) => PySystemError::new_err(msg),
      Shape { .. } => PyException::new_err(msg),
    }
  }
}

impl From<rayon::ThreadPoolBuildError> for LibInflxRsErr {
  fn from(err: rayon::ThreadPoolBuildError) -> Self {
    Self::Rayon(format!("{err}"))
  }
}

/// Signature of gsl_err handler function
pub type GslErrHandler = unsafe extern "C" fn(*const c_char, *const c_char, c_int, c_int) -> !;

#[no_mangle]
pub unsafe extern "C" fn rust_panic_handler(
  reason: *const c_char,
  file: *const c_char,
  lineno: c_int,
  errno: c_int,
) -> ! {
  let reason_str = std::ffi::CStr::from_ptr(reason);
  let file_str = std::ffi::CStr::from_ptr(file);
  let redbold = console::Style::new().red().bold();
  let errcode = redbold.apply_to(format!("(ERRCODE {errno:#00X})"));
  let cyan = console::Style::new().cyan();
  let msg = cyan.apply_to("Error message:");

  println!("{}a GSL exception ocurred {}", *super::PANIC_BADGE, errcode);
  println!("{msg} {reason_str:#?}");
  println!("In {file_str:#?} line number {lineno}");
  panic!();
}
