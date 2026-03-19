use rand::Rng;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub grad: Vec<f32>,
}

// ========================
// Free functions
// ========================

pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0; v.len()];
    let mut sum = 0.0;
    for i in 0..v.len() {
        out[i] = (v[i] - max).exp();
        sum += out[i];
    }
    for i in 0..v.len() {
        out[i] /= sum;
    }
    out
}

pub fn gelu(x: &mut Tensor) {
    for v in &mut x.data {
        let y = *v;
        *v = 0.5 * y * (1.0 + (0.79788456 * (y + 0.044715 * y.powi(3))).tanh());
    }
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.cols, b.rows, "matmul shape mismatch: ({},{}) x ({},{})", a.rows, a.cols, b.rows, b.cols);
    let mut out = Tensor::new(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let a_val = a.data[i * a.cols + k];
            for j in 0..b.cols {
                out.data[i * out.cols + j] += a_val * b.data[k * b.cols + j];
            }
        }
    }
    out
}

pub fn softmax_row(x: &mut Vec<f32>, rows: usize, cols: usize) {
    for i in 0..rows {
        let start = i * cols;
        let max = x[start..start + cols]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for j in 0..cols {
            let val = (x[start + j] - max).exp();
            x[start + j] = val;
            sum += val;
        }
        for j in 0..cols {
            x[start + j] /= sum;
        }
    }
}

pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

// ========================
// Tensor methods
// ========================

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            grad: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_tokens(tokens: &[usize]) -> Tensor {
        let rows = 1;
        let cols = tokens.len();
        let data: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        Tensor {
            rows,
            cols,
            data,
            grad: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (rows + cols) as f32).sqrt(); // Xavier init
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen::<f32>() * 2.0 * scale - scale)
            .collect();
        Self {
            rows,
            cols,
            data,
            grad: vec![0.0; rows * cols],
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            grad: vec![0.0; rows * cols],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![1.0; rows * cols],
            grad: vec![0.0; rows * cols],
        }
    }

    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(a.data.len(), b.data.len(), "add shape mismatch");
        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect();
        Tensor {
            data,
            rows: a.rows,
            cols: a.cols,
            grad: vec![0.0; a.rows * a.cols],
        }
    }

    /// Add bias (1 x cols) to every row of a (rows x cols)
    pub fn add_bias(a: &Tensor, bias: &Tensor) -> Tensor {
        assert_eq!(a.cols, bias.cols);
        let mut out = a.clone();
        for r in 0..a.rows {
            for c in 0..a.cols {
                out.data[r * a.cols + c] += bias.data[c];
            }
        }
        out
    }

    pub fn concat(tensors: &[Tensor]) -> Tensor {
        let rows = tensors[0].rows;
        let mut total_cols = 0;
        for t in tensors {
            assert_eq!(t.rows, rows);
            total_cols += t.cols;
        }
        let mut out = Tensor::zeros(rows, total_cols);
        let mut offset = 0;
        for t in tensors {
            for r in 0..rows {
                for c in 0..t.cols {
                    out.data[r * total_cols + offset + c] = t.data[r * t.cols + c];
                }
            }
            offset += t.cols;
        }
        out
    }

    pub fn transpose(&self) -> Tensor {
        let mut out = Tensor::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * out.cols + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    pub fn last_row(&self) -> Vec<f32> {
        let start = (self.rows - 1) * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// Extract a sub-tensor: all rows, columns [col_start..col_end)
    pub fn slice_cols(&self, col_start: usize, col_end: usize) -> Tensor {
        let new_cols = col_end - col_start;
        let mut out = Tensor::new(self.rows, new_cols);
        for r in 0..self.rows {
            for c in 0..new_cols {
                out.data[r * new_cols + c] = self.data[r * self.cols + col_start + c];
            }
        }
        out
    }

    /// Scale all elements
    pub fn scale(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    /// Zero out the gradient
    pub fn zero_grad(&mut self) {
        for v in &mut self.grad {
            *v = 0.0;
        }
    }
}
