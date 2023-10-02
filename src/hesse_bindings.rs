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

use std::{ffi::OsStr, mem::MaybeUninit};

use ndarray as nd;

use crate::inflatox_version::InflatoxVersion;

type ExFn = unsafe extern "C" fn(*const f64, *const f64) -> f64;
type HdylibFn<'a> = libloading::Symbol<'a, ExFn>;
type HdylibStaticInt<'a> = libloading::Symbol<'a, *const u32>;
type HdyLibStaticArr<'a> = libloading::Symbol<'a, *const [u16; 3]>;

const SYM_DIM_SYM: &[u8; 3] = b"DIM";
const PARAM_DIM_SYM: &[u8; 11] = b"N_PARAMTERS";
const POTENTIAL_SYM: &[u8; 1] = b"V";
const INFLATOX_VERSION_SYM: &[u8; 7] = b"VERSION";

type Error = crate::err::LibInflxRsErr;
type Result<T> = std::result::Result<T, Error>;

pub struct InflatoxDylib {
  lib: libloading::Library,
  n_fields: u32,
  n_param: u32,
  potential: ExFn,
  hesse_cmp: nd::Array2<ExFn>,
  grad_cmp: Vec<ExFn>
}

impl InflatoxDylib {
  /// Try to open an Inflatox compilation artefact at file location `lib_path`.
  /// Returns an error if compilation artefact is incomplete, invalid, or built
  /// for a different inflatox ABI.
  pub(crate) fn open<P: AsRef<OsStr>>(lib_path: P) -> Result<Self> {
    //(0) Convert library path to something more usable
    let libp_string = lib_path.as_ref().to_string_lossy().to_string();

    //(1) Open the compilation artefact
    let lib = unsafe {
      libloading::Library::new(lib_path)
        .map_err(|err| Error::IoErr { lib_path: libp_string.clone(), msg: format!("{err}") })?
    };

    //(2) Check if the artefact is compatible with our version of libinflx
    let inflatox_version = unsafe {
      *lib
        .get::<HdyLibStaticArr>(INFLATOX_VERSION_SYM)
        .map_err(|_err| Error::MissingSymbolErr {
          lib_path: libp_string.clone(),
          symbol: INFLATOX_VERSION_SYM.to_vec(),
        })
        .and_then(|ptr| Ok(**ptr as *mut InflatoxVersion))?
    };
    if inflatox_version != crate::V_INFLX_ABI {
      return Err(Error::VersionErr(inflatox_version));
    }

    //(3) Get number of fields and number of parameters
    let n_fields = unsafe {
      ***lib.get::<HdylibStaticInt>(SYM_DIM_SYM).map_err(|_err| Error::MissingSymbolErr {
        lib_path: libp_string.clone(),
        symbol: SYM_DIM_SYM.to_vec(),
      })?
    };

    let n_param = unsafe {
      ***lib.get::<HdylibStaticInt>(PARAM_DIM_SYM).map_err(|_err| Error::MissingSymbolErr {
        lib_path: libp_string.clone(),
        symbol: PARAM_DIM_SYM.to_vec(),
      })?
    };

    //(4) Get potential hesse, and gradient components
    let potential = unsafe {
      **lib.get::<HdylibFn>(POTENTIAL_SYM).map_err(|_err| Error::MissingSymbolErr {
        lib_path: libp_string.clone(),
        symbol: POTENTIAL_SYM.to_vec(),
      })?
    };
    let hesse_cmp = Self::get_hesse_cmp(&lib, &libp_string, n_fields as usize)?;
    let grad_cmp = Self::get_grad_cmp(&lib, &libp_string, n_fields as usize)?;

    //(R) Return the fully constructed obj
    Ok(InflatoxDylib { lib, n_fields, n_param, potential, hesse_cmp, grad_cmp })
  }

  fn get_hesse_cmp(
    lib: &libloading::Library,
    lib_path: &str,
    n_fields: usize,
  ) -> Result<nd::Array2<ExFn>> {
    let dim = nd::Dim([n_fields, n_fields]);
    let mut array: nd::Array2<MaybeUninit<ExFn>> = nd::Array2::uninit(dim);

    for (idx, uninit) in array.indexed_iter_mut() {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8,
      ];
      let symbol = unsafe {
        **lib.get::<HdylibFn>(raw_symbol).map_err(|_err| Error::MissingSymbolErr {
          lib_path: lib_path.to_string(),
          symbol: raw_symbol.to_vec(),
        })?
      };
      uninit.write(symbol);
    }

    Ok(unsafe { array.assume_init() })
  }

  fn get_grad_cmp(
    lib: &libloading::Library,
    lib_path: &str,
    n_fields: usize,
  ) -> Result<Vec<ExFn>> {
    (0..n_fields)
      .into_iter()
      .map(|idx| unsafe {
        let c = char::from_digit(idx as u32, 10).unwrap() as u32 as u8;
        lib.get::<HdylibFn>(&[b'g', c]).map_err(|_err| Error::MissingSymbolErr {
          lib_path: lib_path.to_string(),
          symbol: vec![b'g', c],
        }).and_then(|x| Ok(**x))
      })
      .collect()
  }

  #[inline(always)]
  /// Calculate scalar potential at field-space coordinates `x`, with model
  /// parameters `p`.
  ///
  /// # Panics
  /// This function panics if `x.len()` does not equal the number of fields of
  /// the loaded model. Similarly, if `p.len()` does not equal the number of
  /// model parameters, this function will panic.
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.n_fields as usize);
    assert!(p.len() == self.n_param as usize);
    unsafe { (self.potential)(x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }

  /// Calculate scalar potential all field-space coordinates in the array `x`,
  /// with model parameters `p`. The physical range that the elements in `x`
  /// represent can be specified by passing the start_stop array.
  ///
  /// # Panics
  /// This function panics if `x.shape().len()` does not equal the number of
  /// fields of the loaded model. Similarly, if `p.len()` does not equal the
  /// number of model parameters, this function will panic.
  pub fn potential_array(&self, mut x: nd::ArrayViewMutD<f64>, p: &[f64], start_stop: &[[f64; 2]]) {
    assert!(x.shape().len() == self.n_fields as usize);
    assert!(p.len() == self.n_param as usize);
    //(1) Convert start-stop ranges
    let (spacings, offsets) = start_stop
      .iter()
      .zip(x.shape().iter())
      .map(|([start, stop], &axis_len)| ((stop - start) / axis_len as f64, *start))
      .unzip::<_, _, Vec<_>, Vec<_>>();
    let mut field_vec = Vec::with_capacity(x.shape().len());

    for (idx, val) in x.indexed_iter_mut() {
      field_vec.clear();
      field_vec.extend(
        (0..self.n_fields as usize).into_iter().map(|i| idx[i] as f64 * spacings[i] + offsets[i]),
      );
      *val = unsafe { (&self.potential)(field_vec.as_ptr(), p.as_ptr()) };
    }
  }

  #[inline(always)]
  /// Calculate projected Hesse matrix at field-space coordinates `x`, with model
  /// parameters `p`.
  ///
  /// # Panics
  /// This function panics if `x.len()` does not equal the number of fields of
  /// the loaded model. Similarly, if `p.len()` does not equal the number of
  /// model parameters, this function will panic.
  pub fn hesse(&self, x: &[f64], p: &[f64]) -> nd::Array2<f64> {
    assert!(x.len() == self.n_fields as usize);
    assert!(p.len() == self.n_param as usize);
    self.hesse_cmp.mapv(|func| unsafe {
      func(x as *const [f64] as *const f64, p as *const [f64] as *const f64)
    })
  }

  #[inline]
  /// Load symbol from underlying dynamic inflatox library.
  pub unsafe fn get_symbol<T>(
    &self,
    symbol: &[u8],
  ) -> std::result::Result<libloading::Symbol<T>, libloading::Error> {
    self.lib.get(symbol)
  }

  #[inline]
  /// Returns number of fields (=dimensionality of the scalar manifold)
  /// of this inflatox model.
  pub const fn get_n_fields(&self) -> usize {
    self.n_fields as usize
  }

  #[inline]
  /// Returns the number of model parameters (excluding fields) of this inflatox
  /// model.
  pub const fn get_n_params(&self) -> usize {
    self.n_param as usize
  }
}

pub struct Hesse2D<'a> {
  lib: &'a InflatoxDylib,
  fns: [ExFn; 4],
}

impl<'a> Hesse2D<'a> {

  pub fn new(lib: &'a InflatoxDylib) -> Self {
    assert!(lib.get_n_fields() == 2);
    let v00 = *lib.hesse_cmp.get((0, 0)).unwrap();
    let v01 = *lib.hesse_cmp.get((1, 0)).unwrap();
    let v10 = *lib.hesse_cmp.get((0, 1)).unwrap();
    let v11 = *lib.hesse_cmp.get((1, 1)).unwrap();
    Hesse2D { lib, fns: [v00, v01, v10, v11] }
  }

  #[inline]
  pub fn v00(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[0](x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }

  #[inline]
  pub fn v01(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[1](x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }

  #[inline]
  pub fn v10(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[2](x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }

  #[inline]
  pub fn v11(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[3](x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }

  #[inline]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    self.lib.potential(x, p)
  }

  #[inline]
  pub const fn get_n_params(&self) -> usize {
    self.lib.get_n_params()
  }
}

pub struct Grad<'a> {
  lib: &'a InflatoxDylib,
  fns: &'a [ExFn]
}

impl<'a> Grad<'a> {

  #[inline]
  pub fn new(lib: &'a InflatoxDylib) -> Self {
    Grad { lib, fns: &lib.grad_cmp }
  }

  #[inline]
  pub fn cmp(&self, x: &[f64], p: &[f64], idx: usize) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[idx](x as *const [f64] as *const f64, p as *const [f64] as *const f64) }
  }
}
