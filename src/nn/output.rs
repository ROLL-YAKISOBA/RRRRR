use crate::tensor::tensor::{Tensor, matmul};
use crate::tensor::softmax::softmax;

/// 出力層: (dim) -> (vocab) の線形変換
pub struct OutputLayer {
    pub weight: Tensor, // rows = dim, cols = vocab
    pub bias: Vec<f32>, // length = vocab
}

impl OutputLayer {
    pub fn new(dim: usize, vocab: usize) -> Self {
        Self {
            weight: Tensor::random(dim, vocab),
            bias: vec![0.0; vocab],
        }
    }

    /// 単一行 (dim,) → logits (vocab,)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.weight.rows;
        let vocab = self.weight.cols;
        assert_eq!(x.len(), dim);

        let mut out = vec![0.0; vocab];
        for v in 0..vocab {
            let mut s = 0.0;
            for d in 0..dim {
                s += x[d] * self.weight.data[d * vocab + v];
            }
            s += self.bias[v];
            out[v] = s;
        }
        out
    }

    /// バッチ forward: (seq x dim) → (seq x vocab)
    pub fn forward_tensor(&self, x: &Tensor) -> Tensor {
        let mut out = matmul(x, &self.weight);
        // add bias to every row
        for r in 0..out.rows {
            for c in 0..out.cols {
                out.data[r * out.cols + c] += self.bias[c];
            }
        }
        out
    }

    /// 単一サンプルの勾配で重みを更新
    pub fn update_from_grad_single(&mut self, hidden: &[f32], grad_logits: &[f32], lr: f32) {
        let dim = self.weight.rows;
        let vocab = self.weight.cols;

        for d in 0..dim {
            for v in 0..vocab {
                let idx = d * vocab + v;
                self.weight.data[idx] -= lr * (hidden[d] * grad_logits[v]);
            }
        }

        for v in 0..vocab {
            self.bias[v] -= lr * grad_logits[v];
        }
    }

    /// バッチ版の重み更新
    pub fn update_from_grad_batch(&mut self, hidden_tensor: &Tensor, grad_logits: &Tensor, lr: f32) {
        let seq = hidden_tensor.rows;
        let dim = hidden_tensor.cols;
        let vocab = grad_logits.cols;

        for i in 0..seq {
            let h_start = i * dim;
            let g_start = i * vocab;
            for d in 0..dim {
                let h = hidden_tensor.data[h_start + d];
                for v in 0..vocab {
                    let idx = d * vocab + v;
                    let g = grad_logits.data[g_start + v];
                    self.weight.data[idx] -= lr * (h * g);
                }
            }
        }

        for i in 0..seq {
            let g_start = i * vocab;
            for v in 0..vocab {
                self.bias[v] -= lr * grad_logits.data[g_start + v];
            }
        }
    }
}