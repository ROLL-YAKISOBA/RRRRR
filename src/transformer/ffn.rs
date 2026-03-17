use crate::tensor::tensor::Tensor;

pub struct FeedForward {

    w1: Tensor,
    w2: Tensor,

}

impl FeedForward {

    pub fn new(dim: usize) -> Self {

        Self {
            w1: Tensor::random(dim, dim * 4),
            w2: Tensor::random(dim * 4, dim),
        }

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        let h = Tensor::matmul(x, &self.w1);
        let h = Tensor::relu(&h);

        Tensor::matmul(&h, &self.w2)

    }

}