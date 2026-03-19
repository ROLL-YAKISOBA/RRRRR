use crate::transformer::multihead::MultiHeadAttention;
use crate::nn::feedforward::FeedForward;
use crate::tensor::tensor::Tensor;
use crate::transformer::layernorm::LayerNorm;

pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(dim: usize, n_heads: usize) -> Self {
        let hidden = dim * 4; // 標準的なFFN隠れ層サイズ

        Self {
            attn: MultiHeadAttention::new(dim, n_heads),
            ffn: FeedForward::new(dim, hidden),
            norm1: LayerNorm::new(dim),
            norm2: LayerNorm::new(dim),
        }
    }

    /// Pre-norm Transformer block
    /// x: (seq x dim) → (seq x dim)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // --- Self Attention with residual ---
        let x1 = self.norm1.forward(x);
        let attn = self.attn.forward(&x1);
        let x2 = Tensor::add(x, &attn);

        // --- Feed Forward with residual ---
        let x3 = self.norm2.forward(&x2);
        let ff = self.ffn.forward(&x3);

        Tensor::add(&x2, &ff)
    }
}