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