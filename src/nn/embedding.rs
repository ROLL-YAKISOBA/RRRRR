use crate::tensor::tensor::Tensor;
use crate::nn::position::positional_encoding;

pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        Self {
            weight: Tensor::random(vocab_size, dim),
        }
    }

    /// tokens → (seq_len x dim) embedding + positional encoding
    pub fn forward(&self, tokens: &[usize]) -> Tensor {
        let seq_len = tokens.len();
        let dim = self.weight.cols;

        let mut out = Vec::with_capacity(seq_len * dim);

        for &t in tokens {
            assert!(t < self.weight.rows, "token {} out of vocab range {}", t, self.weight.rows);
            let start = t * dim;
            let end = start + dim;
            out.extend_from_slice(&self.weight.data[start..end]);
        }

        let token_emb = Tensor {
            data: out,
            rows: seq_len,
            cols: dim,
            grad: vec![0.0; seq_len * dim],
        };

        let pos = positional_encoding(seq_len, dim);
        Tensor::add(&token_emb, &pos)
    }
}
