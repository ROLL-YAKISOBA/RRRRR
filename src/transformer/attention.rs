use crate::tensor::tensor::{Tensor, matmul, softmax_row};
use crate::transformer::mask::causal_mask;

pub struct Attention {

    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,

}

impl Attention {

    pub fn new(d: usize) -> Self {

        Self {
            wq: Tensor::random(d, d),
            wk: Tensor::random(d, d),
            wv: Tensor::random(d, d),
        }

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        let q = matmul(x, &self.wq);
        let k = matmul(x, &self.wk);
        let v = matmul(x, &self.wv);

        let kt = Tensor::transpose(&k);

        let mut scores = matmul(&q, &kt);

        let scale = (x.cols as f32).sqrt();

        for s in &mut scores.data {
            *s /= scale;
        }

        let mask = causal_mask(scores.rows);

        for i in 0..scores.data.len() {
            scores.data[i] += mask.data[i];
        }

        softmax_row(&mut scores.data, scores.rows, scores.cols);

        matmul(&scores, &v)
    }
}
/* 
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
*/