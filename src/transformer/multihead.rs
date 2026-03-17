use crate::tensor::tensor::Tensor;
//use crate::transformer::attention::SelfAttention;
use crate::transformer::attention::Attention;

pub struct MultiHeadAttention {

    heads: Vec<Attention>,
    wo: Tensor,

}

impl MultiHeadAttention {

     pub fn new(dim: usize, n_heads: usize) -> Self {

        let mut heads = Vec::new();

        for _ in 0..n_heads {
            heads.push(Attention::new(dim));
        }

        Self {
            heads,
            wo: Tensor::random(dim * n_heads, dim)
        }
    }


    pub fn forward(&self, x: &Tensor) -> Tensor {

        let mut outputs = Vec::new();

        for head in &self.heads {
            outputs.push(head.forward(x));
        }

        let concat = Tensor::concat(&outputs);

        Tensor::matmul(&concat, &self.wo)
    }
}