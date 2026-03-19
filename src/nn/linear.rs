use crate::tensor::tensor::{Tensor, matmul};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            weight: Tensor::random(input, output),
            bias: Tensor::zeros(1, output),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = matmul(x, &self.weight);
        Tensor::add_bias(&out, &self.bias)
    }
}