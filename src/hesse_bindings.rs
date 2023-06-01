use std::{ffi::OsStr, mem::MaybeUninit};

use ndarray as nd;

type HdylibFn<'a> = libloading::Symbol::<'a, unsafe extern fn (*const f64, *const f64) -> f64>;
type HdylibStaticInt<'a> = libloading::Symbol::<'a, *const u32>;
type HdyLibStaticArr<'a> = libloading::Symbol<'a, *const [u16; 3]>;

const SYM_DIM_SYM: &[u8; 3] = b"DIM";
const PARAM_DIM_SYM: &[u8; 11] = b"N_PARAMTERS";
const POTENTIAL_SYM: &[u8; 1] = b"V";
const INFLATOX_VERSION_SYM: &[u8; 7] = b"VERSION";

pub struct InflatoxDylib {
  lib: libloading::Library,
  f_dim: u32,
  n_param: u32,
  inflatox_version: [u16; 3]
}

impl InflatoxDylib {
  pub fn new<P: AsRef<OsStr>>(path: P) -> Result<Self, libloading::Error> {
    let lib = unsafe { libloading::Library::new(path)? };
    let f_dim = unsafe { ***lib.get::<HdylibStaticInt>(SYM_DIM_SYM)? };
    let n_param = unsafe { ***lib.get::<HdylibStaticInt>(PARAM_DIM_SYM)? };
    let inflatox_version = unsafe { ***lib.get::<HdyLibStaticArr>(INFLATOX_VERSION_SYM)? };
    Ok( InflatoxDylib { lib, f_dim, n_param, inflatox_version })
  }

  #[inline]
  pub unsafe fn get_symbol<T>(&self, symbol: &[u8]) -> Result<libloading::Symbol<T>, libloading::Error> {
    self.lib.get(symbol)
  }

  #[inline]
  pub const fn get_dim(&self) -> usize {
    self.f_dim as usize
  }

  #[inline]
  pub const fn get_n_params(&self) -> usize {
    self.n_param as usize
  }

  #[inline]
  pub const fn get_inflatox_version(&self) -> &[u16; 3] {
    &self.inflatox_version
  }
}

pub struct HesseNd<'a> {
  lib: &'a InflatoxDylib,
  potential: unsafe extern fn (*const f64, *const f64) -> f64,
  components: nd::Array2<HdylibFn<'a>>
}

impl<'a>HesseNd<'a> {
  pub fn new(lib: &'a InflatoxDylib) -> Result<Self, libloading::Error> {
    let dim = nd::Dim([lib.get_dim(), lib.get_dim()]);
    let mut array: nd::Array2<MaybeUninit<HdylibFn>> = nd::Array2::uninit(dim);
    let potential = unsafe { **lib.get_symbol::<HdylibFn>(POTENTIAL_SYM).unwrap() };

    array.indexed_iter_mut().for_each(|(idx, uninit)| {
      let raw_symbol = &[
        b'v',
        char::from_digit(idx.0 as u32, 10).unwrap() as u32 as u8,
        char::from_digit(idx.1 as u32, 10).unwrap() as u32 as u8
      ];
      let symbol: HdylibFn = unsafe { lib.get_symbol(raw_symbol).unwrap() };
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
  potential: unsafe extern fn (*const f64, *const f64) -> f64,
  fns: [unsafe extern fn (*const f64, *const f64) -> f64; 4]
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