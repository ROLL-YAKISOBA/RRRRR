use crate::tensor::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl LayerNorm {

    pub fn new(dim: usize) -> Self {

        Self {
            gamma: Tensor::random(1, dim),
            beta: Tensor::zeros(1, dim),
        }

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        let mut out = x.clone();

        for r in 0..x.rows {

            let start = r * x.cols;
            let end = start + x.cols;

            let row = &x.data[start..end];

            let mean = row.iter().sum::<f32>() / x.cols as f32;

            let var = row.iter()
                .map(|v| (v-mean)*(v-mean))
                .sum::<f32>() / x.cols as f32;

            let std = (var + 1e-5).sqrt();

            for c in 0..x.cols {

                let idx = start + c;

                out.data[idx] =
                    (x.data[idx] - mean) / std
                    * self.gamma.data[c]
                    + self.beta.data[c];
            }

        }

        out
    }
}
/* 
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

*/
                