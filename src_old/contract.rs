//! This file defines tensor contractions using strict Einstein notation
//! (no index repeated more than once!)

use ndarray as nd;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Represents Covariant and Contravariant tensorial indices
enum Idx {
  Cov(usize),
  Con(usize)
}

struct Tensor<T, D: nd::Dimension> {
  indices: Vec<Idx>,
  data: nd::Array<T, D>
}

impl<T, D> std::ops::Add for Tensor<T, D>
where
  T: std::ops::Add<Output = T> + Clone + num_traits::Zero,
  D: nd::Dimension
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    if self.indices != rhs.indices {
      panic!("Cannot add tensors of different rank together!")
    }
    let mut out = nd::Array::zeros(self.data.raw_dim());
    self
      .data
      .view()
      .iter()
      .zip(rhs.data.view().iter())
      .zip(out.iter_mut())
      .for_each(|((&x, &y), &mut out)| out = x + y);

    Tensor {
      indices: self.indices.clone(), data: out
    }
  }
}

trait Contract {
  type Output;
  fn contract(&self, idxs: &[(Idx, Idx)]) -> Self::Output;
}

impl<T,D> Contract for Tensor<T, D>
where
  T: std::ops::Add<Output = T> + Clone + num_traits::Zero,
  D: nd::Dimension,

trait EinsteinSum {
  type Output;
  fn contract(&self, other: &Self, idxs: &[(Idx, Idx)]) -> Self::Output;
}

impl<T, D, E> EinsteinSum for Tensor<T, D>
where
  T: std::ops::Add<Output = T> + Clone + num_traits::Zero,
  D: nd::Dimension,
  E: nd::Dimension
{
  type Output = Tensor<T, E>;

  fn contract(&self, other: &Self, idxs: &[(Idx, Idx)]) -> Self::Output {
    let out_dim: isize =
      self.data.shape().len() as isize +
      other.data.shape().len() as isize + 
      -2 * idxs.len() as isize;
    if out_dim < 0 {
      panic!("Cannot contract more indices than the rank of the tensor product!")
    }
    let out_shape = vec![self.data.shape()[0]; out_dim as usize];
    let mut out = nd::linalg::Dot()

    todo!()
  }
}