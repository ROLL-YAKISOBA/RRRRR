use crate::nn::embedding::Embedding;
use crate::transformer::block::TransformerBlock;
use crate::transformer::layernorm::LayerNorm;
use crate::nn::output::OutputLayer;
use crate::tensor::tensor::Tensor;

pub struct GPT {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub output: OutputLayer,
    pub vocab: usize,
    pub dim: usize,
}

impl GPT {
    /// vocab: 語彙サイズ, n_layer: Transformer layers, dim: 埋め込み次元, n_heads: attention heads
    pub fn new(vocab: usize, n_layer: usize, dim: usize, n_heads: usize) -> Self {
        let embedding = Embedding::new(vocab, dim);

        let mut blocks = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            blocks.push(TransformerBlock::new(dim, n_heads));
        }

        let final_norm = LayerNorm::new(dim);
        let output = OutputLayer::new(dim, vocab);

        GPT {
            embedding,
            blocks,
            final_norm,
            output,
            vocab,
            dim,
        }
    }

    /// tokens → hidden representation (seq x dim)
    pub fn encode(&self, tokens: &[usize]) -> Tensor {
        // Embedding already includes positional encoding
        let mut x = self.embedding.forward(tokens);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final layer norm
        self.final_norm.forward(&x)
    }

    /// tokens → logits (seq x vocab)
    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let hidden = self.encode(tokens);
        self.output.forward_tensor(&hidden)
    }
}