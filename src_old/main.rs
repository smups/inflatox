use nd::{array, ShapeBuilder};
use ndarray as nd;
use ndarray_linalg::*;

mod contract;

fn main() {
    println!("ĲNDELĲK")
}


struct Tensor {
    dim: usize,
    rank: (usize, usize),
    components: nd::ArrayD<f64>,
}

impl Tensor {
    pub fn zeros(dim: usize, rank: (usize, usize)) -> Self {
        Tensor {
            dim,
            rank,
            components: nd::ArrayD::zeros(vec![dim; rank.0 + rank.1].into_shape()),
        }
    }
    pub fn ones(dim: usize, rank: (usize, usize)) -> Self {
        Tensor {
            dim,
            rank,
            components: nd::ArrayD::ones(vec![dim; rank.0 + rank.1].into_shape()),
        }
    }

    pub fn from_array(
        array: nd::ArrayD<f64>,
        rank: (usize, usize),
    ) -> Result<Self, TryFromArrayErr> {
        if array.shape().len() != rank.0 + rank.1 {
            return Err(TryFromArrayErr::InvalidRank {
                array_rank: array.shape().len(),
                rank,
            });
        }

        let dim = array.shape()[0];
        if !array.shape().iter().all(|&idx| idx == dim) {
            return Err(TryFromArrayErr::NonSquareArray(array.shape().to_vec()));
        }

        Ok(Tensor {
            dim: array.shape().len(),
            rank,
            components: array,
        })
    }

    /// Computes outer product of the two tensors. If self is a rank (a, b) tensor
    /// and other a rank (c, d) tensor, then the output tensor will be of rank
    /// (a+c, b+d)
    pub fn out_prod(&self, other: &Tensor) -> Tensor {
        assert!(self.dim == other.dim);
        //Alloc output tensor
        let mut output = nd::Array::zeros(vec![self.dim; self.rank.0 + self.rank.1 + other.rank.0 + other.rank.1]);

        //Fill output tensor
        output.indexed_iter_mut().for_each(|(dim, &mut val)| {
            let shape = dim.ix;
        });
    }

    pub fn contract(&self, other: &Tensor, ax1: usize, ax2: usize) -> Tensor {

    }
}

enum ContractErr {

}

#[derive(Debug, Clone)]
enum TryFromArrayErr {
    InvalidRank { array_rank: usize, rank: (usize, usize) },
    NonSquareArray(Vec<usize>),
}
impl std::error::Error for TryFromArrayErr {}
impl std::fmt::Display for TryFromArrayErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TryFromArrayErr::*;
        match self {
            InvalidRank { array_rank, rank } =>
                write!(f, "Array with {array_rank} ax(e/i)s cannot be cast to a tensor of rank {rank:?}"),
            NonSquareArray(shape) => 
                write!(f, "Array with non-square shape {shape:?} cannot be cast to a tensor")
        }
    }
}

trait SpaceTimeMetric {
    const DIM: usize;

    fn cmp(&self, event: &[f64]) -> nd::Array2<f64>;
    fn inv(&self, event: &[f64]) -> nd::Array2<f64> {
        let g = self.cmp(event);
        g.inv().expect("non-invertable metric?")
    }
}

/// Components are in the following order:
/// tt, xx, yy, zz, tx, ty, tz, xy, xz, yz
struct GenericSpaceTimeMetric {
    components: [fn(&[f64]) -> f64; 10],
}

impl SpaceTimeMetric for GenericSpaceTimeMetric {
    const DIM: usize = 4;
    fn cmp(&self, event: &[f64]) -> nd::Array2<f64> {
        //Compute components of g
        let mut g: [f64; 10] = [0.0; 10];
        for (funcs, g_ij) in self.components.iter().zip(g.iter_mut()) {
            *g_ij = funcs(event)
        }

        //Set components of output matrix
        array![
            [g[0], g[1], g[2], g[3]],
            [g[1], g[4], g[5], g[6]],
            [g[2], g[5], g[7], g[8]],
            [g[3], g[6], g[8], g[9]]
        ]
    }
}

struct FLRWMetric {
    scale_factor: fn(f64) -> f64,
    curvature: FLRWCurvature,
}

enum FLRWCurvature {
    Open,
    Closed,
    Flat,
}

impl SpaceTimeMetric for FLRWMetric {
    const DIM: usize = 4;
    fn cmp(&self, event: &[f64]) -> nd::Array2<f64> {
        let a2 = (self.scale_factor)(event[0]).pow(2.0);
        array![
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, a2, 0.0, 0.0],
            [0.0, 0.0, a2, 0.0],
            [0.0, 0.0, 0.0, a2]
        ]
    }
}

#[test]
fn inflation() {
    let inflation_metric = FLRWMetric {
        scale_factor: |t| f64::exp(t),
        curvature: FLRWCurvature::Flat,
    };
    let minkowski = MinkowskiMetric::new();
    assert_eq!(inflation_metric.cmp(&[0.0]), minkowski.cmp(&[0.0]));
    let eye = inflation_metric
        .cmp(&[50.0])
        .apply2(&inflation_metric.inv(&[50.0]));
    assert!(eye == nd::Array2::eye(4));
}

struct MinkowskiMetric {
    components: nd::Array2<f64>,
}

impl SpaceTimeMetric for MinkowskiMetric {
    const DIM: usize = 4;
    fn cmp(&self, _event: &[f64]) -> nd::Array2<f64> {
        self.components.clone()
    }
    fn inv(&self, _event: &[f64]) -> nd::Array2<f64> {
        self.components.clone()
    }
}

impl MinkowskiMetric {
    pub fn new() -> Self {
        MinkowskiMetric {
            components: array![
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
        }
    }
}

#[test]
fn test_minkowski_metric() {
    let minkowski = MinkowskiMetric::new();
    let minkowski_inv = minkowski.inv(&[0.0]);
    let minkowski = minkowski.cmp(&[0.0]);
    let eye = minkowski.apply2(&minkowski_inv);
    println!("{eye:?}");
    assert!(eye == nd::Array2::eye(4));
}
