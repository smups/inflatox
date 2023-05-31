mod hesse_bindings;
use hesse_bindings::*;

fn main() {
  let lib = HesseDylib::new("/tmp/libinflx_autoc_pz0lpkp_.so").unwrap();
  println!("{}-{}", lib.get_dim(), lib.get_n_params());
  let hesse_2d = Hesse2D::new(&lib).unwrap();
  let v00 = hesse_2d.v00(&[1.0, 2.0], &[10.0, 20.0, 30.0]);
  println!("v00(1,2) with p = [10, 20, 30] equals: {v00}")  
}