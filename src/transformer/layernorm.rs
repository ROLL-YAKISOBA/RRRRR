use crate::tensor::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub dim: usize,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            dim,
        }
    }

    /// x: (rows x dim) → (rows x dim) via layer normalization
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.cols, self.dim);
        let mut out = x.clone();

        for r in 0..x.rows {
            let start = r * x.cols;

            // mean
            let mut mean = 0.0;
            for c in 0..x.cols {
                mean += x.data[start + c];
            }
            mean /= x.cols as f32;

            // variance
            let mut var = 0.0;
            for c in 0..x.cols {
                let diff = x.data[start + c] - mean;
                var += diff * diff;
            }
            var /= x.cols as f32;

            let std = (var + 1e-5).sqrt();

            // normalize + scale/shift
            for c in 0..x.cols {
                let idx = start + c;
                out.data[idx] = (x.data[idx] - mean) / std * self.gamma[c] + self.beta[c];
            }
        }

        out
    }
}