use std::ffi::OsStr;

type HdylibFn<'a> = libloading::Symbol::<'a, unsafe extern fn (*const f64, *const f64) -> f64>;
type HdylibStatic<'a> = libloading::Symbol::<'a, *const u32>;

const SYM_DIM: &[u8; 3] = b"DIM";
const PARAM_DIM: &[u8; 11] = b"N_PARAMTERS";

pub struct HesseDylib {
  lib: libloading::Library,
  f_dim: u32,
  n_param: u32
}

impl HesseDylib {
  pub fn new<P: AsRef<OsStr>>(path: P) -> Result<Self, libloading::Error> {
    let lib = unsafe {
      libloading::Library::new(path)?
    };
    let hesse_dim: u32 = unsafe {
      let symbol: HdylibStatic = lib.get(SYM_DIM)?;
      **symbol
    };
    let n_param: u32 = unsafe {
      let symbol: HdylibStatic = lib.get(PARAM_DIM)?;
      **symbol
    };
    Ok( HesseDylib { lib, f_dim: hesse_dim, n_param })
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
}

pub struct Hesse2D<'a> {
  lib: &'a HesseDylib,
  fns: [HdylibFn<'a>; 4]
}

impl<'a> Hesse2D<'a> {
  pub fn new(lib: &'a HesseDylib) -> Result<Self, libloading::Error> {
    unsafe {
      let v00: HdylibFn = lib.get_symbol(b"v00")?;
      let v01: HdylibFn = lib.get_symbol(b"v01")?;
      let v10: HdylibFn = lib.get_symbol(b"v10")?;
      let v11: HdylibFn = lib.get_symbol(b"v11")?;

      Ok(Hesse2D { lib, fns: [v00, v01, v10, v11] })
    }
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