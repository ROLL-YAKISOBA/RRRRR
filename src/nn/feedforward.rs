use crate::tensor::tensor::{Tensor, matmul, gelu};

pub struct FeedForward {

    pub w1: Tensor,
    pub w2: Tensor,

}

impl FeedForward {

    pub fn new(d: usize, hidden: usize) -> Self {

        Self {
            w1: Tensor::random(d, hidden),
            w2: Tensor::random(hidden, d),
        }

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        let mut h = matmul(x, &self.w1);

        gelu(&mut h);

        matmul(&h, &self.w2)
    }
}

/* 
use crate::tensor::tensor::*;

pub struct FeedForward {
    pub w1: Tensor,
    pub w2: Tensor,
    pub dim: usize,
    pub hidden: usize,
}

impl FeedForward {

    pub fn new(dim: usize) -> Self {

        let hidden = dim * 4;

        Self {
            w1: Tensor::new(dim, hidden),
            w2: Tensor::new(hidden, dim),
            dim,
            hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        // x * w1
        let mut out = Tensor::matmul(x, &self.w1);

        // activation
        relu(&mut out);

        // hidden → dim
        Tensor::matmul(&out, &self.w2)
    }
}

*/