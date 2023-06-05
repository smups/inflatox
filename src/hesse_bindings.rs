use std::{ffi::OsStr, mem::MaybeUninit, ops::Deref};

use ndarray as nd;
use pyo3::{prelude::*, exceptions::PyIOError};

use crate::inflatox_version::InflatoxVersion;

type ExFn = unsafe extern fn (*const f64, *const f64) -> f64;
type HdylibFn<'a> = libloading::Symbol::<'a, ExFn>;
type HdylibStaticInt<'a> = libloading::Symbol::<'a, *const u32>;
type HdyLibStaticArr<'a> = libloading::Symbol<'a, *const [u16; 3]>;

const SYM_DIM_SYM: &[u8; 3] = b"DIM";
const PARAM_DIM_SYM: &[u8; 11] = b"N_PARAMTERS";
const POTENTIAL_SYM: &[u8; 1] = b"V";
const INFLATOX_VERSION_SYM: &[u8; 7] = b"VERSION";

pub struct InflatoxDylib {
  lib: libloading::Library,
  n_fields: u32,
  n_param: u32,
  inflatox_version: InflatoxVersion
}

#[pyclass]
/// Python wrapper for `InflatoxDyLib`
struct InflatoxPyDyLib {
  inner: InflatoxDylib
}

#[pyfunction]
fn open_inflx_dylib(lib_path: &str) -> PyResult<InflatoxPyDyLib> {
  //(1) Open the compilation artefact
  let lib = unsafe {
    libloading::Library::new(lib_path)
      .map_err(|err| PyIOError::new_err(format!("Could not load Inflatox Compilation Artefact (path: {lib_path}). Error: \"{err}\"")))?
  };

  //(2) Get number of fields, parameters and the inflatox version this artefact
  //was compiled for
  let n_fields = unsafe { ***lib.get::<HdylibStaticInt>(SYM_DIM_SYM)
    .map_err(|err| pyo3::exceptions::PySystemError::new_err(format!("Could not find symbol {SYM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\"")))?
  };
  let n_param = unsafe { ***lib.get::<HdylibStaticInt>(PARAM_DIM_SYM)
    .map_err(|err| pyo3::exceptions::PySystemError::new_err(format!("Could not find symbol {PARAM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\"")))?
  };
  let inflatox_version = unsafe { *lib.get::<HdyLibStaticArr>(INFLATOX_VERSION_SYM)
    .map_err(|err| pyo3::exceptions::PySystemError::new_err(format!("Could not find symbol {INFLATOX_VERSION_SYM:#?} in {lib_path}. Error: \"{err}\"")))
    .and_then(|ptr| Ok(**ptr as *mut InflatoxVersion))?
  };

  //(3) Check that the artefact was built with the correct version of inflatox
  if inflatox_version != super::V_INFLX {
    Err(pyo3::exceptions::PySystemError::new_err(format!("Cannot load Inflatox Compilation Artefact compiled for Inflatox {inflatox_version} using current Inflatox installation ({})", super::V_INFLX)))
  } else {
    Ok(InflatoxPyDyLib { inner: InflatoxDylib { lib, n_fields, n_param, inflatox_version } })
  }
}

impl InflatoxDylib {

  #[inline]
  pub unsafe fn get_symbol<T>(&self, symbol: &[u8]) -> Result<libloading::Symbol<T>, libloading::Error> {
    self.lib.get(symbol)
  }

  #[inline]
  pub const fn get_dim(&self) -> usize {
    self.n_fields as usize
  }

  #[inline]
  pub const fn get_n_params(&self) -> usize {
    self.n_param as usize
  }

  #[inline]
  pub fn get_inflatox_version(&self) -> String {
    format!("{}", self.inflatox_version)
  }

  #[inline]
  pub(crate) const fn get_inflatox_version_raw(&self) -> &InflatoxVersion {
    &self.inflatox_version
  }
}

pub struct HesseNd<'a> {
  lib: &'a InflatoxDylib,
  potential: ExFn,
  components: nd::Array2<ExFn>
}

impl<'a>HesseNd<'a> {
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self, libloading::Error> {
    let dim = nd::Dim([lib.get_dim(), lib.get_dim()]);
    let mut array: nd::Array2<MaybeUninit<ExFn>> = nd::Array2::uninit(dim);
    let potential = unsafe { **lib.get_symbol::<HdylibFn>(POTENTIAL_SYM).unwrap() };

    array.indexed_iter_mut().for_each(|(idx, uninit)| {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8
      ];
      let symbol = unsafe { **lib.get_symbol::<HdylibFn>(raw_symbol).unwrap() };
      uninit.write(symbol);
    });

    Ok(HesseNd { lib, potential, components: unsafe { array.assume_init() } })
  }

  #[inline(always)]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { (self.potential)(
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  pub fn hesse(&self, x: &[f64], p: &[f64]) -> nd::Array2<f64> {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    self.components.mapv(|func| unsafe {
      func(x as *const [f64] as *const f64, p as *const [f64] as *const f64)
    })
  }
}

pub struct Hesse2D<'a> {
  lib: &'a InflatoxDylib,
  potential: ExFn,
  fns: [ExFn; 4]
}

impl<'a> Hesse2D<'a> {
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self, libloading::Error> {
    unsafe {
      let v00 = **lib.get_symbol::<HdylibFn>(b"v00")?;
      let v01 = **lib.get_symbol::<HdylibFn>(b"v01")?;
      let v10 = **lib.get_symbol::<HdylibFn>(b"v10")?;
      let v11 = **lib.get_symbol::<HdylibFn>(b"v11")?;
      let potential = **lib.get_symbol::<HdylibFn>(POTENTIAL_SYM).unwrap();

      Ok(Hesse2D { lib, potential, fns: [v00, v01, v10, v11] })
    }
  }

  #[inline(always)]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { (self.potential)(
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v00(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[0](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v01(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[1](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v10(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[2](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v11(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_dim());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[3](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }
}