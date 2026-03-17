use crate::transformer::mask::causal_mask;
use crate::nn::linear::Linear;
use crate::tensor::tensor::Tensor;

use crate::tensor::softmax::softmax;
//pub struct Attention
pub struct SelfAttention {

    wq: Linear,
    wk: Linear,
    wv: Linear

}

impl SelfAttention {

    pub fn new(dim: usize) -> Self {

        Self {

            wq: Linear::new(dim, dim),
            wk: Linear::new(dim, dim),
            wv: Linear::new(dim, dim)

        }

    }

pub fn forward(&self, x: &Tensor) -> Tensor {

    let q = self.wq.forward(x);
    let k = self.wk.forward(x);
    let v = self.wv.forward(x);

    let kt = transpose(&k);

    let mut scores = Tensor::matmul(&q, &kt);

    // scale
    let scale = (q.cols as f32).sqrt();
    for s in &mut scores.data {
        *s /= scale;
    }

    // causal mask
    let mask = causal_mask(scores.rows);

    for i in 0..scores.data.len() {
        scores.data[i] += mask.data[i];
    }

    let probs = Tensor::softmax_tensor(&scores);

    Tensor::matmul(&probs, &v)
}

}

fn transpose(t: &Tensor) -> Tensor {

    let mut result = Tensor::new(t.cols, t.rows);

    for i in 0..t.rows {
        for j in 0..t.cols {

            result.data[j*t.rows + i] =
                t.data[i*t.cols + j];

        }
    }

    result

}