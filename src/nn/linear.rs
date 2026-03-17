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