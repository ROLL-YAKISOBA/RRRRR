use crate::tensor::tensor::Tensor;

/// 出力層（線形）：W: (dim x vocab), optional bias b: (vocab)
pub struct OutputLayer {
    pub weight: Tensor,   // rows = dim, cols = vocab
    pub bias: Option<Vec<f32>>,
}

impl OutputLayer {
    pub fn new(dim: usize, vocab: usize) -> Self {
        // Tensor::random(rows, cols) を用意している想定
        Self {
            weight: Tensor::random(dim, vocab),
            bias: Some(vec![0.0; vocab]),
        }
    }

    /// 単一行（1 x dim）に対する forward（既存の API と互換）
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x: len == dim
        let dim = self.weight.rows;
        let vocab = self.weight.cols;
        assert_eq!(x.len(), dim);

        let mut out = vec![0.0; vocab];
        for v in 0..vocab {
            let mut s = 0.0;
            for d in 0..dim {
                s += x[d] * self.weight.data[d * vocab + v]; // weight stored row-major: rows*cols
            }
            if let Some(b) = &self.bias {
                s += b[v];
            }
            out[v] = s;
        }
        out
    }

    /// バッチ一括 forward: X (seq x dim) -> logits (seq x vocab) を返す
    pub fn forward_tensor(&self, x: &Tensor) -> Tensor {
        // 標準の行列積: (seq x dim) * (dim x vocab) = (seq x vocab)
        crate::tensor::ops::matmul(x, &self.weight)
        // bias はここでは付加しないが必要なら Tensor::add で行追加する実装を加えてください
    }

    /// 出力層の重みを更新するヘルパー
    /// hidden: hidden row (dim) , grad_logits: grad for logits (vocab), lr: learning rate
    /// ここでは bias も更新する
    pub fn update_from_grad_single(&mut self, hidden: &[f32], grad_logits: &[f32], lr: f32) {
        let dim = self.weight.rows;
        let vocab = self.weight.cols;
        assert_eq!(hidden.len(), dim);
        assert_eq!(grad_logits.len(), vocab);

        // dW_{d,v} += hidden[d] * grad_logits[v]  (we accumulate and then subtract lr*dW)
        for d in 0..dim {
            for v in 0..vocab {
                let idx = d * vocab + v;
                self.weight.data[idx] -= lr * (hidden[d] * grad_logits[v]);
            }
        }

        if let Some(b) = &mut self.bias {
            for v in 0..vocab {
                b[v] -= lr * grad_logits[v];
            }
        }
    }

    /// バッチ版の重み更新: hidden_tensor (seq x dim), grad_logits_tensor (seq x vocab), lr
    /// 実装は per-row update の合計と等価
    pub fn update_from_grad_batch(&mut self, hidden_tensor: &Tensor, grad_logits: &Tensor, lr: f32) {
        let seq = hidden_tensor.rows;
        let dim = hidden_tensor.cols;
        let gv = grad_logits.cols; // vocab
        assert_eq!(grad_logits.rows, seq);
        assert_eq!(dim, self.weight.rows);
        assert_eq!(gv, self.weight.cols);

        for i in 0..seq {
            let h_start = i * dim;
            let g_start = i * gv;
            for d in 0..dim {
                let h = hidden_tensor.data[h_start + d];
                for v in 0..gv {
                    let idx = d * gv + v;
                    let g = grad_logits.data[g_start + v];
                    self.weight.data[idx] -= lr * (h * g);
                }
            }
        }

        if let Some(b) = &mut self.bias {
            for i in 0..seq {
                let g_start = i * gv;
                for v in 0..gv {
                    b[v] -= lr * grad_logits.data[g_start + v];
                }
            }
        }
    }
}

/*
pub struct OutputLayer {

    pub weight: Vec<f32>,
    pub dim: usize,
    pub vocab: usize,

}

impl OutputLayer {

    pub fn new(dim: usize, vocab: usize) -> Self {

        Self {

            weight: vec![0.01; dim * vocab],
            dim,
            vocab,

        }

    }

}



impl OutputLayer {

   pub fn forward(&self, x: &[f32]) -> Vec<f32> {

        let mut logits = vec![0.0; self.vocab];

        for v in 0..self.vocab {

            let mut sum = 0.0;

            for d in 0..self.dim {

                sum += x[d] * self.weight[d*self.vocab + v];

            }

            logits[v] = sum;

        }

        logits

    }

}

*/