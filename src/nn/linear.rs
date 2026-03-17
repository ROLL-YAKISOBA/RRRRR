//use crate::tensor::Tensor;
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

        let mut out = matmul(x, &self.weight);
        out = Tensor::add(&out, &self.bias);

        out
    }
}

/*
use crate::tensor::tensor::Tensor;
use crate::tensor::ops::matmul;
use rand;

pub struct Linear {
    pub weight: Tensor
}

impl Linear {

    pub fn new(in_dim: usize, out_dim: usize) -> Self {

        let mut data = vec![0.0; out_dim * in_dim];

        for i in 0..data.len() {
            data[i] = rand::random::<f32>() * 0.01;
        }

        Self {
            weight: Tensor {
                data,
                rows: in_dim,
                cols: out_dim
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        matmul(x, &self.weight)
    }

}
*/