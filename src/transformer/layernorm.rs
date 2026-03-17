
use crate::tensor::tensor::Tensor;

pub struct LayerNorm {

    dim: usize,
    eps: f32,

}

impl LayerNorm {

    pub fn new(dim: usize) -> Self {

        Self {
            dim,
            eps: 1e-5
        }

    }


    pub fn forward(&self, x: &Tensor) -> Tensor {

    assert_eq!(x.cols, self.dim);

    let mut out = x.clone();

    for i in 0..x.rows {

        let start = i * x.cols;
        let end = start + x.cols;

        let mut mean = 0.0;

        for j in start..end {
            mean += x.data[j];
        }

        mean /= x.cols as f32;

        let mut var = 0.0;

        for j in start..end {
            var += (x.data[j] - mean).powi(2);
        }

        var /= x.cols as f32;

        let std = (var + self.eps).sqrt();

        for j in start..end {
            out.data[j] = (x.data[j] - mean) / std;
        }

    }

    out
}

}
                