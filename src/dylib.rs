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

use ndarray as nd;
use std::{
  ffi::{c_char, OsStr},
  io::Write,
  mem::MaybeUninit,
  path::PathBuf,
};

use crate::inflatox_version::InflatoxVersion;

const INFLATOX_VERSION_SYM: &[u8; 7] = b"VERSION";
const MODEL_NAME_SYM: &[u8; 10] = b"MODEL_NAME";
const USE_GSL_SYM: &[u8; 7] = b"USE_GSL";
const DIM_SYM: &[u8; 3] = b"DIM";
const PARAM_DIM_SYM: &[u8; 12] = b"N_PARAMETERS";
const POTENTIAL_SYM: &[u8; 1] = b"V";
const METRIC_SYM: &[u8; 10] = b"inner_prod";
const GRADIENT_SQUARE_SYM: &[u8; 17] = b"grad_norm_squared";
const EOM_SYM: &[u8; 3] = b"eom";
const HUBBLE_CONSTRAINT_SYM: &[u8; 4] = b"eomh";
const GSL_INIT_SYM: &[u8; 9] = b"err_setup";

pub type ExFn2 = unsafe extern "C" fn(*const f64, *const f64) -> f64;
pub type ExFn3 = unsafe extern "C" fn(*const f64, *const f64, *const f64) -> f64;
pub type ExFn4 = unsafe extern "C" fn(*const f64, *const f64, *const f64, *const f64) -> f64;
pub type ExVecFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64) -> std::ffi::c_void;
type ExInitFn = unsafe extern "C" fn(crate::err::GslErrHandler);

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

pub struct InflatoxDylib {
  dylib_handle: libloading::Library,
  model_name: String,
  path: PathBuf,
  n_fields: u32,
  n_param: u32,
  potential: ExFn2,
  grad_square: ExFn2,
}

impl InflatoxDylib {
  /// Try to open an Inflatox compilation artefact at file location `lib_path`.
  /// Returns an error if compilation artefact is incomplete, invalid, or built
  /// for a different inflatox ABI.
  pub fn open<P: AsRef<OsStr>>(ref lib_path: P) -> Result<Self> {
    // Convert library path to something more usable
    let libp_string = lib_path.as_ref().to_string_lossy().to_string();

    // Open the compilation artefact
    let lib = unsafe {
      libloading::Library::new(lib_path)
        .map_err(|err| Error::Io { lib_path: libp_string.clone(), msg: format!("{err}") })?
    };

    // Check if the artefact is compatible with our version of libinflx
    let inflatox_version = unsafe {
      lib
        .get::<*const [u16; 3]>(INFLATOX_VERSION_SYM)
        .map_err(|_err| Error::MissingSymbol {
          lib_path: libp_string.clone(),
          symbol: INFLATOX_VERSION_SYM.to_vec(),
        })
        .map(|inner| std::mem::transmute::<_, *const InflatoxVersion>(inner))?
    };
    // Why not cast the crate::V_INFLX_ABI to a *const? Unlike when comparing references, comparing
    // pointers does not compare the values behind the pointers, but the pointers themselves, which
    // is not what we want here
    if &unsafe { *inflatox_version } != &crate::V_INFLX_ABI {
      return Err(Error::Version(unsafe { *inflatox_version }));
    }

    // Get number of fields and number of parameters
    let n_fields = unsafe {
      **lib.get::<*const u32>(DIM_SYM).map_err(|_err| Error::MissingSymbol {
        lib_path: libp_string.clone(),
        symbol: DIM_SYM.to_vec(),
      })?
    };

    let n_param = unsafe {
      **lib.get::<*const u32>(PARAM_DIM_SYM).map_err(|_err| Error::MissingSymbol {
        lib_path: libp_string.clone(),
        symbol: PARAM_DIM_SYM.to_vec(),
      })?
    };

    // Parse model name
    let mname_raw = unsafe {
      let mname_ptr = *lib.get::<*const c_char>(MODEL_NAME_SYM).map_err(|_err| {
        Error::MissingSymbol { lib_path: libp_string.clone(), symbol: MODEL_NAME_SYM.to_vec() }
      })?;
      std::ffi::CStr::from_ptr(mname_ptr)
    };
    let model_name = mname_raw.to_string_lossy().to_string();

    // Get potential hesse, and gradient components
    let potential = unsafe {
      *lib.get::<ExFn2>(POTENTIAL_SYM).map_err(|_err| Error::MissingSymbol {
        lib_path: libp_string.clone(),
        symbol: POTENTIAL_SYM.to_vec(),
      })?
    };

    // Get the size of the gradient squared (special quantity)
    let grad_square = unsafe {
      *lib.get::<ExFn2>(GRADIENT_SQUARE_SYM).map_err(|_err| Error::MissingSymbol {
        lib_path: libp_string.clone(),
        symbol: GRADIENT_SQUARE_SYM.to_vec(),
      })?
    };

    // Check if the library uses the GSL, and initialise if this is the case
    let gsl: c_char = unsafe {
      **lib.get::<*const c_char>(USE_GSL_SYM).map_err(|_err| Error::MissingSymbol {
        lib_path: libp_string.clone(),
        symbol: MODEL_NAME_SYM.to_vec(),
      })?
    };
    if gsl == 1 {
      let gsl_init_fn = unsafe {
        *lib.get::<ExInitFn>(GSL_INIT_SYM).map_err(|_err| Error::MissingSymbol {
          lib_path: libp_string.clone(),
          symbol: MODEL_NAME_SYM.to_vec(),
        })?
      };
      unsafe { gsl_init_fn(crate::err::rust_panic_handler) }
    }

    // Return the fully constructed obj
    Ok(InflatoxDylib {
      dylib_handle: lib,
      model_name,
      path: PathBuf::from(lib_path),
      n_fields,
      n_param,
      potential,
      grad_square,
    })
  }

  pub fn get_hesse_cmp(&self) -> Result<nd::Array2<ExFn2>> {
    let dim = nd::Dim([self.n_fields as usize, self.n_fields as usize]);
    let mut array: nd::Array2<MaybeUninit<ExFn2>> = nd::Array2::uninit(dim);

    for (idx, uninit) in array.indexed_iter_mut() {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8,
      ];
      let symbol = unsafe {
        *self.dylib_handle.get::<ExFn2>(raw_symbol).map_err(|_err| Error::MissingSymbol {
          lib_path: self.path.to_string_lossy().into_owned(),
          symbol: raw_symbol.to_vec(),
        })?
      };
      uninit.write(symbol);
    }

    Ok(unsafe { array.assume_init() })
  }

  pub fn get_basis_fn(&self, idx: u32) -> Result<ExVecFn> {
    if idx > self.n_fields {
      return Err(Error::FieldDim {
        expected: self.n_fields,
        got: idx,
        msg: "number of field-space basis vectors is limited to the number of scalar fields"
          .to_string(),
      });
    }

    if idx == 0 {
      unsafe { self.dylib_handle.get::<ExVecFn>(b"v") }
        .map_err(|_err| Error::MissingSymbol {
          symbol: vec![b"v"[0]],
          lib_path: self.path.to_string_lossy().into_owned(),
        })
        .map(|x| *x)
    } else {
      let mut symbol = Vec::new();
      symbol.write_fmt(format_args!("w{idx}")).unwrap();
      unsafe { self.dylib_handle.get::<ExVecFn>(&symbol) }
        .map_err(|_err| Error::MissingSymbol {
          symbol,
          lib_path: self.path.to_string_lossy().into_owned(),
        })
        .map(|x| *x)
    }
  }

  pub fn get_basis_fns(&self) -> Result<Box<[ExVecFn]>> {
    (0..self.n_fields())
      .map(|fidx| {
        if fidx == 0 {
          unsafe { self.dylib_handle.get::<ExVecFn>(b"v") }
            .map_err(|_err| Error::MissingSymbol {
              symbol: vec![b"v"[0]],
              lib_path: self.path.to_string_lossy().into_owned(),
            })
            .map(|x| *x)
        } else {
          let mut symbol = Vec::with_capacity(3);
          symbol.write_fmt(format_args!("w{fidx}")).unwrap();
          unsafe {
            self
              .dylib_handle
              .get::<ExVecFn>(&symbol)
              .map_err(|_err| Error::MissingSymbol {
                symbol,
                lib_path: self.path.to_string_lossy().into_owned(),
              })
              .map(|x| *x)
          }
        }
      })
      .collect()
  }

  pub fn get_inner_prod_fn(&self) -> Result<ExFn4> {
    unsafe {
      self
        .dylib_handle
        .get::<ExFn4>(METRIC_SYM)
        .map_err(|_err| Error::MissingSymbol {
          symbol: METRIC_SYM.to_vec(),
          lib_path: self.path.to_string_lossy().into_owned(),
        })
        .map(|x| *x)
    }
  }

  pub fn get_eom(&self) -> Result<Box<[ExFn3]>> {
    (0..self.n_fields())
      .map(|fidx| unsafe {
        let mut symbol = Vec::with_capacity(EOM_SYM.len());
        symbol.extend_from_slice(EOM_SYM);
        symbol.extend_from_slice(format!("{fidx}").as_bytes());
        self
          .dylib_handle
          .get::<ExFn3>(&symbol)
          .map_err(|_err| Error::MissingSymbol {
            symbol,
            lib_path: self.path.to_string_lossy().into_owned(),
          })
          .map(|x| *x)
      })
      .collect()
  }

  pub fn get_hubble_constraint(&self) -> Result<ExFn3> {
    unsafe {
      self
        .dylib_handle
        .get::<ExFn3>(HUBBLE_CONSTRAINT_SYM)
        .map_err(|_err| Error::MissingSymbol {
          symbol: HUBBLE_CONSTRAINT_SYM.to_vec(),
          lib_path: self.path.to_string_lossy().into_owned(),
        })
        .map(|x| *x)
    }
  }

  #[inline(always)]
  pub const fn grad_square(&self) -> ExFn2 {
    self.grad_square
  }

  #[inline(always)]
  pub const fn potential(&self) -> ExFn2 {
    self.potential
  }

  #[inline(always)]
  /// returns the model name
  pub fn name(&self) -> &str {
    &self.model_name
  }

  #[inline(always)]
  /// Load symbol from underlying dynamic inflatox library.
  pub unsafe fn get_symbol<T>(
    &self,
    symbol: &[u8],
  ) -> std::result::Result<libloading::Symbol<T>, libloading::Error> {
    self.dylib_handle.get(symbol)
  }

  #[inline(always)]
  /// Returns number of fields (=dimensionality of the scalar manifold)
  /// of this inflatox model.
  pub const fn n_fields(&self) -> usize {
    self.n_fields as usize
  }

  #[inline(always)]
  /// Returns the number of model parameters (excluding fields) of this inflatox
  /// model.
  pub const fn n_pars(&self) -> usize {
    self.n_param as usize
  }
}
