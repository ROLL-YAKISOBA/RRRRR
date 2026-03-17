use crate::tensor::tensor::Tensor;

pub fn softmax(t: &Tensor) -> Tensor {

    let mut out = t.data.clone();

    for r in 0..t.rows {

        let start = r * t.cols;
        let end = start + t.cols;

        let row = &t.data[start..end];

        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0;

        for i in 0..t.cols {
            out[start + i] = (row[i] - max).exp();
            sum += out[start + i];
        }

        for i in 0..t.cols {
            out[start + i] /= sum;
        }
    }

    Tensor {
        data: out,
        rows: t.rows,
        cols: t.cols
    }
}