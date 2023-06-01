mod hesse_bindings;
use hesse_bindings::*;

fn main() {
  let lib = HesseDylib::new("/tmp/libinflx_autoc_9u8fmfgo.so").unwrap();
  println!("{}-{}", lib.get_dim(), lib.get_n_params());
  let hesse_2d = Hesse2D::new(&lib).unwrap();
  let v00 = hesse_2d.v00(&[1.0, 2.0], &[10.0, 20.0, 30.0]);
  println!("v00(1,2) with p = [10, 20, 30] equals: {v00}");

  let args = &[10.0, 50.0, 1.0, 1.5, 3.0];
  test_3d(&[100.0, 500.0, 10.0], args);
  /*
    V0 -> args[0]
    m -> args[1]
    φ0 -> args[2]
    θ0 -> args[3]
    ψ0 -> args[4] 
  */
}

fn test_3d(x: &[f64], p: &[f64]) {
  let lib = HesseDylib::new("/tmp/libinflx_autoc_oz8kke7_.so").unwrap();
  let hesse_nd = HesseNd::new(&lib).unwrap();
  let h = hesse_nd.calc_hesse(x, p);
  println!("{h:?}");
}