mod hesse_bindings;
use hesse_bindings::*;

use ndarray as nd;
use rayon::prelude::*;

fn main() {
  let lib = InflatoxDylib::new("/tmp/libinflx_autoc_d9w6fwe4.so").unwrap();
  println!("{}-{}", lib.get_dim(), lib.get_n_params());
  let hesse_2d = Hesse2D::new(&lib).unwrap();
  let p = &[10.0, 20.0, 30.0, 10.0];
  let v00 = hesse_2d.v00(&[1.0, 2.0], &[10.0, 20.0, 30.0, 10.0]);
  println!("v00(1,2) with p = [10, 20, 30] equals: {v00}");

  calc_anguelova(&hesse_2d, p);
}

fn test_3d(x: &[f64], p: &[f64]) {
  let lib = InflatoxDylib::new("/tmp/libinflx_autoc_oz8kke7_.so").unwrap();
  let hesse_nd = HesseNd::new(&lib).unwrap();
  let h = hesse_nd.hesse(x, p);
  println!("{h:?}");
}

#[inline]
fn anguelova(h: &Hesse2D, x: &[f64], p: &[f64]) -> f64 {
  let lhs = 3.0 * h.potential(x, p) * (h.v00(x, p) / h.v01(x, p).powi(2)).powi(2);
  let rhs = h.v11(x, p);
  return lhs - rhs;
}

fn calc_anguelova(h: &Hesse2D, p: &[f64]) {
  let size = 10_000;
  let grid_spacing = 1E-2;
  let offst = (size / 2) as f64 * -grid_spacing;

  let field_space = nd::Array2::from_shape_fn([size, size], |idx| 
    [idx.0 as f64 * grid_spacing + offst, idx.1 as f64 * grid_spacing + offst]
  );
  let mut out = nd::Array::zeros([size, size]);

  nd::Zip::from(out.view_mut())
    .and(field_space.view())
    .into_par_iter()
    .for_each(|(val, x)| {
      *val = anguelova(h, x, p)
    });
  
  println!("{out:?}");
}