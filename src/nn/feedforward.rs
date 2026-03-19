use crate::tensor::tensor::{Tensor, matmul, gelu};

pub struct FeedForward {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
}

impl FeedForward {
    pub fn new(d: usize, hidden: usize) -> Self {
        Self {
            w1: Tensor::random(d, hidden),
            b1: Tensor::zeros(1, hidden),
            w2: Tensor::random(hidden, d),
            b2: Tensor::zeros(1, d),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = matmul(x, &self.w1);
        h = Tensor::add_bias(&h, &self.b1);
        gelu(&mut h);
        let mut out = matmul(&h, &self.w2);
        out = Tensor::add_bias(&out, &self.b2);
        out
    }
}