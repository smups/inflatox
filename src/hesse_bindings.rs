use std::mem::MaybeUninit;

use ndarray as nd;
use pyo3::{prelude::*, exceptions::{PyIOError, PySystemError}};
#[cfg(feature = "pyo3_extension_module")]
use numpy::PyReadonlyArray1;

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
  potential: ExFn,
  inflatox_version: InflatoxVersion,
}

#[pyclass]
/// Python wrapper for `InflatoxDyLib`
pub(crate) struct InflatoxPyDyLib(pub InflatoxDylib);

#[cfg(feature = "pyo3_extension_module")]
#[pymethods]
impl InflatoxPyDyLib {
  fn potential(
    &self,
    x: PyReadonlyArray1<f64>,
    p : PyReadonlyArray1<f64>
  ) -> PyResult<f64> {
    //(0) Convert the PyArrays to nd::Arrays
    let p = p.as_array();
    let x = x.as_array();

    //(3) Make sure that the number of supplied fields matches the number
    //specified by the dynamic lib
    if x.shape() != &[self.0.n_fields as usize] {
      raise_shape_err(format!("expected a {}D array as field-space coordinate. Found array with shape {:?}", self.0.n_fields, x.shape()))?;
    }
    let x = x.as_slice().unwrap();

    //(3) Make sure that the number of supplied model parameters matches the number
    //specified by the dynamic lib
    if p.shape() != &[self.0.n_param as usize] {
      raise_shape_err(format!("expected a {}D as parameters set. Found array with shape {:?}", self.0.n_param, p.shape()))?;
    }
    let p = p.as_slice().unwrap();

    //(4) Calculate
    Ok(self.0.potential(x, p))
  }
}

#[pyfunction]
pub(crate) fn open_inflx_dylib(lib_path: &str) -> PyResult<InflatoxPyDyLib> {
  Ok(InflatoxPyDyLib(InflatoxDylib::open(lib_path)?))
}

pub(crate) fn raise_shape_err<T>(err: String) -> PyResult<T> {
  Err(super::ShapeError::new_err(err))
}

pub(crate) fn convert_start_stop(
  start_stop: nd::ArrayView2<f64>,
  n_fields: usize
) -> PyResult<Vec<[f64; 2]>> {
  if start_stop.shape().len() != 2
  || start_stop.shape()[1] != n_fields
  || start_stop.shape()[0] != 2
  {
    raise_shape_err(format!("start_stop array should have 2 rows and as many columns as there are fields ({}). Got start_stop with shape {:?}", n_fields, start_stop.shape()))?;
  }
  let start_stop = start_stop
    .axis_iter(nd::Axis(0))
    .map(|start_stop| [start_stop[0], start_stop[1]])
    .collect::<Vec<_>>();
  Ok(start_stop)
}

impl InflatoxDylib {

  pub(crate) fn open(lib_path: &str) -> PyResult<Self> {
    //(1) Open the compilation artefact
    let lib = unsafe {
      libloading::Library::new(lib_path)
        .map_err(|err| PyIOError::new_err(format!("Could not load Inflatox Compilation Artefact (path: {lib_path}). Error: \"{err}\"")))?
    };

    //(2) Get number of fields, parameters and the inflatox version this artefact
    //was compiled for
    let n_fields = unsafe { ***lib.get::<HdylibStaticInt>(SYM_DIM_SYM)
      .map_err(|err| PySystemError::new_err(format!("Could not find symbol {SYM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\"")))?
    };
    let n_param = unsafe { ***lib.get::<HdylibStaticInt>(PARAM_DIM_SYM)
      .map_err(|err| PySystemError::new_err(format!("Could not find symbol {PARAM_DIM_SYM:#?} in {lib_path}. Error: \"{err}\"")))?
    };
    let inflatox_version = unsafe { *lib.get::<HdyLibStaticArr>(INFLATOX_VERSION_SYM)
      .map_err(|err| PySystemError::new_err(format!("Could not find symbol {INFLATOX_VERSION_SYM:#?} in {lib_path}. Error: \"{err}\"")))
      .and_then(|ptr| Ok(**ptr as *mut InflatoxVersion))?
    };
    let potential = unsafe { **lib.get::<HdylibFn>(POTENTIAL_SYM)
      .map_err(|err| PySystemError::new_err(format!("Could not find symbol {POTENTIAL_SYM:#?} in {lib_path}. Error: \"{err}\"")))?
    };

    //(3) Check that the artefact was built with the correct version of inflatox
    if inflatox_version != super::V_INFLX {
      return Err(PySystemError::new_err(format!("Cannot load Inflatox Compilation Artefact compiled for Inflatox {inflatox_version} using current Inflatox installation ({})", super::V_INFLX)))
    } else {
      Ok(InflatoxDylib { lib, n_fields, n_param, potential, inflatox_version } )
    }
  }

  #[inline(always)]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.n_fields as usize);
    assert!(p.len() == self.n_param as usize);
    unsafe { (self.potential)(
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub unsafe fn get_symbol<T>(&self, symbol: &[u8]) -> Result<libloading::Symbol<T>, libloading::Error> {
    self.lib.get(symbol)
  }

  #[inline]
  pub const fn get_n_fields(&self) -> usize {
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
  components: nd::Array2<ExFn>
}

impl<'a> HesseNd<'a> {
  pub fn new(lib: &'a InflatoxDylib) -> Self {
    let dim = nd::Dim([lib.get_n_fields(), lib.get_n_fields()]);
    let mut array: nd::Array2<MaybeUninit<ExFn>> = nd::Array2::uninit(dim);

    array.indexed_iter_mut().for_each(|(idx, uninit)| {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8
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
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 { self.lib.potential(x, p) }
  #[inline]
  pub const fn get_n_fields(&self) -> usize { self.lib.get_n_fields() }
  #[inline]
  pub const fn get_n_params(&self) -> usize { self.lib.get_n_params() }
  #[inline]
  pub fn get_inflatox_version(&self) -> String { self.lib.get_inflatox_version() }
  #[inline]
  pub(crate) const fn get_inflatox_version_raw(&self) -> &InflatoxVersion {
    self.lib.get_inflatox_version_raw()
  }

}

pub struct Hesse2D<'a> {
  lib: &'a InflatoxDylib,
  fns: [ExFn; 4]
}

impl<'a> Hesse2D<'a> {
  pub fn new(nd: HesseNd<'a>) -> Self {
    assert!(nd.lib.get_n_fields() == 2);
    let v00 = *nd.components.get((0,0)).unwrap();
    let v01 = *nd.components.get((1,0)).unwrap();
    let v10 = *nd.components.get((0,1)).unwrap();
    let v11 = *nd.components.get((1,1)).unwrap();
    Hesse2D { lib: nd.lib, fns: [v00, v01, v10, v11] }
  }

  #[inline]
  pub fn v00(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[0](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v01(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[1](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v10(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[2](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn v11(&self, x: &[f64], p: &[f64]) -> f64 {
    assert!(x.len() == self.lib.get_n_fields());
    assert!(p.len() == self.lib.get_n_params());
    unsafe { self.fns[3](
      x as *const [f64] as *const f64,
      p as *const [f64] as *const f64
    )}
  }

  #[inline]
  pub fn potential(&self, x: &[f64], p: &[f64]) -> f64 { self.lib.potential(x, p) }
  #[inline]
  pub const fn get_n_fields(&self) -> usize { self.lib.get_n_fields() }
  #[inline]
  pub const fn get_n_params(&self) -> usize { self.lib.get_n_params() }
  #[inline]
  pub fn get_inflatox_version(&self) -> String { self.lib.get_inflatox_version() }
  #[inline]
  pub(crate) const fn get_inflatox_version_raw(&self) -> &InflatoxVersion {
    self.lib.get_inflatox_version_raw()
  }
}