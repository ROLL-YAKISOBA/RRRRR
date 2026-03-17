
use crate::tensor::tensor::Tensor;

pub fn cross_entropy(logits: &Tensor, targets: &Vec<usize>) -> f32 {

    let mut loss = 0.0;

    for i in 0..targets.len() {

        let start = i * logits.cols;
        let end = start + logits.cols;

        let row = &logits.data[start..end];

        let mut max = f32::NEG_INFINITY;
        for v in row {
            if *v > max {
                max = *v;
            }
        }

        let mut sum = 0.0;

        for v in row {
            sum += (*v - max).exp();
        }

        let log_prob = (row[targets[i]] - max).exp() / sum;

        loss -= log_prob.ln();
    }

    loss / targets.len() as f32
}