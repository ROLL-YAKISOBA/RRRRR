
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub fn softmax(v: &[f32]) -> Vec<f32> {

    let mut out = vec![0.0; v.len()];

    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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

/*
pub fn softmax(v: &Vec<f32>) -> Vec<f32> {

    let mut exp_values = Vec::new();
    let mut sum = 0.0;

    for x in v {
        let e = x.exp();
        exp_values.push(e);
        sum += e;
    }

    exp_values.iter().map(|x| x / sum).collect()

}

    */

impl Tensor {

    pub fn new(rows: usize, cols: usize) -> Self {

        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }

    }

    
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {

    assert_eq!(a.cols, b.rows);

    let mut out = Tensor::new(a.rows, b.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {

            let mut sum = 0.0;

            for k in 0..a.cols {

                sum +=
                    a.data[i*a.cols + k]
                    *
                    b.data[k*b.cols + j];

            }

            out.data[i*out.cols + j] = sum;

        }
    }

    out
}

  
  pub fn random(rows: usize, cols: usize) -> Self {

    use rand::Rng;

    let mut rng = rand::thread_rng();

    let mut data = vec![0.0; rows * cols];

    for i in 0..data.len() {
        data[i] = rng.gen::<f32>() * 0.02 - 0.01;
    }

    Self { rows, cols, data }
   }


pub fn zeros(rows: usize, cols: usize) -> Self {

    Self {
        rows,
        cols,
        data: vec![0.0; rows * cols]
    }

}


    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {

    let mut data = vec![0.0; a.rows * a.cols];

    for i in 0..data.len() {
        data[i] = a.data[i] + b.data[i];
    }

    Tensor {
        data,
        rows: a.rows,
        cols: a.cols,
    }

}

pub fn concat(tensors: &Vec<Tensor>) -> Tensor {

    let rows = tensors[0].rows;

    let mut cols = 0;

    for t in tensors {
        cols += t.cols;
    }

    let mut out = Tensor::zeros(rows, cols);

    let mut offset = 0;

    for t in tensors {

        for r in 0..rows {
            for c in 0..t.cols {

                out.data[r*cols + offset + c] =
                    t.data[r*t.cols + c];

            }
        }

        offset += t.cols;
    }

    out
}


    pub fn matmul_attn(scores: &Vec<f32>, v: &Tensor, n: usize, d: usize) -> Vec<f32> {

    let mut out = vec![0.0; n * d];

    for i in 0..n {
        for j in 0..n {
            let s = scores[i*n + j];

            for k in 0..d {
                out[i*d + k] += s * v.data[j*d + k];
            }
        }
    }

    out
}




pub fn softmax_tensor(x: &Tensor) -> Tensor {

    let mut out = x.clone();

    let max = out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0;

    for v in &mut out.data {
        *v = (*v - max).exp();
        sum += *v;
    }

    for v in &mut out.data {
        *v /= sum;
    }

    out
}

pub fn apply_temperature(logits: &mut Vec<f32>, temperature: f32) {

    for v in logits.iter_mut() {
        *v /= temperature;
    }

}


pub fn softmax(x: & Vec<f32>) ->  Vec<f32> {

    let mut max = f32::MIN;

    for v in x {
        if *v > max {
            max = *v;
        }
    }

    let mut exps = Vec::with_capacity(x.len());
    let mut sum = 0.0;

    for v in x {
        let e = (*v - max).exp();
        exps.push(e);
        sum += e;
    }

    for v in &mut exps {
        *v /= sum;
    }

    exps
}

}

/* 
pub fn softmax(x: &Vec<f32>) -> Vec<f32> {

    let mut max = f32::MIN;

    for v in x {
        if *v > max {
            max = *v;
        }
    }

    let mut exps = Vec::with_capacity(x.len());
    let mut sum = 0.0;

    for v in x {
        let e = (*v - max).exp();
        exps.push(e);
        sum += e;
    }

    for v in &mut exps {
        *v /= sum;
    }

    exps
}
*/

pub fn relu(x: &mut Tensor) {

    for v in &mut x.data {

        if *v < 0.0 {
            *v = 0.0;
        }

    }

}


pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {

    assert_eq!(a.cols, b.rows);

    let mut out = Tensor::new(a.rows, b.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {

            let mut sum = 0.0;

            for k in 0..a.cols {

                sum +=
                    a.data[i*a.cols + k]
                    *
                    b.data[k*b.cols + j];

            }

            out.data[i*out.cols + j] = sum;

        }
    }

    out
}

/* 
pub fn matmul(
    x: &Vec<f32>,
    w: &Vec<f32>,
    n: usize,
    d: usize,
) -> Vec<f32> {

    let mut out = vec![0.0; n * d];

    for i in 0..n {
        for j in 0..d {
            let mut sum = 0.0;

            for k in 0..d {
                sum += x[i*d + k] * w[k*d + j];
            }

            out[i*d + j] = sum;
        }
    }

    out
}
    */



    pub fn softmax_row(x: &mut Vec<f32>, rows: usize, cols: usize) {

    for i in 0..rows {

        let start = i * cols;
        let end = start + cols;

        let row = &x[start..end];

        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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
    



/* 
  */
  



pub fn add(a: &Tensor, b: &Tensor) -> Tensor {

    let mut data = vec![0.0; a.rows * a.cols];

    for i in 0..data.len() {
        data[i] = a.data[i] + b.data[i];
    }

    Tensor {
        data,
        rows: a.rows,
        cols: a.cols,
    }

}


