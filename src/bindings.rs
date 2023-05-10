include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use ndarray as nd;
use libloading;
use ndarray_linalg::Inverse;

struct SpaceTimeMetric<'lib>{
  lib: &'lib libloading::Library
}

impl SpaceTimeMetric {

  pub fn metric(event: &[f64; 4]) -> nd::Array2<f64> {
    let ptr = event as *const f64;
    unsafe { nd::array![
      [g00(ptr), g01(ptr), g02(ptr), g03(ptr)],
      [g01(ptr), g11(ptr), g12(ptr), g13(ptr)],
      [g02(ptr), g12(ptr), g22(ptr), g23(ptr)],
      [g03(ptr), g13(ptr), g23(ptr), g33(ptr)]
    ]}
  }

  pub fn inverse_metric(event: &[f64; 4]) -> nd::Array2<f64> {
    Self::metric(event).inv().expect("non-invertable metric!")
  }
}

macro_rules! metric_components {
  ($cmp:ident) => {
    fn $cmp(&self, x: &[f64; 4]) -> f64 {
      type cmp = unsafe extern fn(*const f64) -> f64;
    }
  };
}