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

use std::mem::MaybeUninit;

use ndarray as nd;
#[cfg(feature = "pyo3_extension_module")]
use numpy::{PyArray2, PyReadonlyArrayDyn};
use pyo3::{
  exceptions::{PyIOError, PySystemError},
  prelude::*,
};

use crate::inflatox_version::InflatoxVersion;

type ExFn = unsafe extern fn (*const f64, *const f64) -> f64;
type HdylibFn<'a> = libloading::Symbol<'a, ExFn>;
type HdylibStaticInt<'a> = libloading::Symbol<'a, *const u32>;
type HdyLibStaticArr<'a> = libloading::Symbol<'a, *const [u16; 3]>;

const SYM_DIM_SYM: &[u8; 3] = b"DIM";
const PARAM_DIM_SYM: &[u8; 11] = b"N_PARAMTERS";
const POTENTIAL_SYM: &[u8; 1] = b"V";
const INFLATOX_VERSION_SYM: &[u8; 7] = b"VERSION";

pub struct InflatoxDylib {
  lib: libloading::Library,
  n_fields: u32,
  n_param: u32,
  potential: ExFn
}

impl InflatoxDylib {

  /// Try to open an Inflatox compilation artefact at file location `lib_path`.
  /// Returns an error if compilation artefact is incomplete, invalid, or built
  /// for a different inflatox ABI.
  pub(crate) fn open(lib_path: &str) -> PyResult<Self> {
    //(1) Open the compilation artefact
    let lib = unsafe {
      libloading::Library::new(lib_path).map_err(|err| {
        PyIOError::new_err(format!(
          "Could not load Inflatox Compilation Artefact (path: {lib_path}). Error: \"{err}\""
        ))
      })?
    };

    //(2) Get number of fields, parameters and the inflatox version this artefact
    //was compiled for
    let n_fields = unsafe {
      ***lib.get::<HdylibStaticInt>(SYM_DIM_SYM).map_err(|err| {
        PySystemError::new_err(format!(
          "Could not find symbol {SYM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\""
        ))
      })?
    };
    let n_param = unsafe {
      ***lib.get::<HdylibStaticInt>(PARAM_DIM_SYM).map_err(|err| {
        PySystemError::new_err(format!(
          "Could not find symbol {PARAM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\""
        ))
      })?
    };
    let inflatox_version = unsafe {
      *lib
        .get::<HdyLibStaticArr>(INFLATOX_VERSION_SYM)
        .map_err(|err| {
          PySystemError::new_err(format!(
            "Could not find symbol {INFLATOX_VERSION_SYM:#?} in {lib_path}. Error: \"{err}\""
          ))
        })
        .and_then(|ptr| Ok(**ptr as *mut InflatoxVersion))?
    };
    let potential = unsafe {
      **lib.get::<HdylibFn>(POTENTIAL_SYM).map_err(|err| {
        PySystemError::new_err(format!(
          "Could not find symbol {POTENTIAL_SYM:#?} in {lib_path}. Error: \"{err}\""
        ))
      })?
    };

    //(3) Check that the artefact was built with the correct version of inflatox
    if inflatox_version != super::V_INFLX_ABI {
      return Err(PySystemError::new_err(format!("Cannot load Inflatox Compilation Artefact compiled for Inflatox {inflatox_version} using current Inflatox installation ({})", super::V_INFLX_ABI)));
    } else {
      Ok(InflatoxDylib { lib, n_fields, n_param, potential })
    }
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

  #[inline]
  /// Load symbol from underlying dynamic inflatox library.
  pub unsafe fn get_symbol<T>(
    &self,
    symbol: &[u8],
  ) -> Result<libloading::Symbol<T>, libloading::Error> {
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

pub struct HesseNd<'a> {
  lib: &'a InflatoxDylib,
  components: nd::Array2<ExFn>,
}

impl<'a> HesseNd<'a> {
  pub fn new(lib: &'a InflatoxDylib) -> Self {
    let dim = nd::Dim([lib.get_n_fields(), lib.get_n_fields()]);
    let mut array: nd::Array2<MaybeUninit<ExFn>> = nd::Array2::uninit(dim);

    array.indexed_iter_mut().for_each(|(idx, uninit)| {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8,
      ];
      let symbol = unsafe { **lib.get_symbol::<HdylibFn>(raw_symbol).unwrap() };
      uninit.write(symbol);
    });

    HesseNd { lib, components: unsafe { array.assume_init() } }
  }

  pub fn hesse(&self, x: &[f64], p: &[f64]) -> nd::Array2<f64> {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    self.components.mapv(|func| unsafe {
      func(x as *const [f64] as *const f64, p as *const [f64] as *const f64)
    })
  }

  #[inline]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    self.lib.potential(x, p)
  }

  #[inline]
  pub const fn get_n_fields(&self) -> usize {
    self.lib.get_n_fields()
  }

  #[inline]
  pub const fn get_n_params(&self) -> usize {
    self.lib.get_n_params()
  }

}

pub struct Hesse2D<'a> {
  lib: &'a InflatoxDylib,
  fns: [ExFn; 4],
}

impl<'a> Hesse2D<'a> {
  pub fn new(nd: HesseNd<'a>) -> Self {
    assert!(nd.lib.get_n_fields() == 2);
    let v00 = *nd.components.get((0, 0)).unwrap();
    let v01 = *nd.components.get((1, 0)).unwrap();
    let v10 = *nd.components.get((0, 1)).unwrap();
    let v11 = *nd.components.get((1, 1)).unwrap();
    Hesse2D { lib: nd.lib, fns: [v00, v01, v10, v11] }
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
  pub const fn get_n_fields(&self) -> usize {
    self.lib.get_n_fields()
  }
  
  #[inline]
  pub const fn get_n_params(&self) -> usize {
    self.lib.get_n_params()
  }

}
