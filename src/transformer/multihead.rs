use crate::tensor::tensor::{Tensor, matmul};
use crate::transformer::attention::Attention;

/// MultiHead Attention: n_heads 個のヘッドを並列実行し、concat → Wo
pub struct MultiHeadAttention {
    pub heads: Vec<Attention>,
    pub wo: Tensor, // (dim x dim) — concat結果(n_heads * head_dim = dim)からdimへ
}

impl MultiHeadAttention {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        assert_eq!(dim % n_heads, 0, "dim must be divisible by n_heads");
        let head_dim = dim / n_heads;

        let mut heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            heads.push(Attention::new(dim, head_dim));
        }

        Self {
            heads,
            wo: Tensor::random(dim, dim), // concat output (dim) → dim
        }
    }

    /// x: (seq x dim) → (seq x dim)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let outputs: Vec<Tensor> = self.heads.iter().map(|h| h.forward(x)).collect();
        let concat = Tensor::concat(&outputs); // (seq x dim)
        matmul(&concat, &self.wo) // (seq x dim)
    }
}