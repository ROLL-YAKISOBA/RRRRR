use crate::tensor::tensor::{Tensor, matmul, softmax_row};
use crate::transformer::mask::causal_mask;

/// 単一ヘッドの Self-Attention
/// head_dim = dim / n_heads として使う
pub struct Attention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(dim: usize, head_dim: usize) -> Self {
        Self {
            wq: Tensor::random(dim, head_dim),
            wk: Tensor::random(dim, head_dim),
            wv: Tensor::random(dim, head_dim),
            head_dim,
        }
    }

    /// x: (seq x dim) → output: (seq x head_dim)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = matmul(x, &self.wq); // (seq x head_dim)
        let k = matmul(x, &self.wk); // (seq x head_dim)
        let v = matmul(x, &self.wv); // (seq x head_dim)

        let kt = k.transpose(); // (head_dim x seq)

        let mut scores = matmul(&q, &kt); // (seq x seq)

        let scale = (self.head_dim as f32).sqrt();
        for s in &mut scores.data {
            *s /= scale;
        }

        // causal mask
        let mask = causal_mask(scores.rows);
        for i in 0..scores.data.len() {
            scores.data[i] += mask.data[i];
        }

        softmax_row(&mut scores.data, scores.rows, scores.cols);

        matmul(&scores, &v) // (seq x head_dim)
    }
}