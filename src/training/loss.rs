use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::softmax;


pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &[usize],
) -> f32 {

    let mut loss = 0.0;

    for i in 0..targets.len() {

        let start = i * logits.cols;
        let end = start + logits.cols;

        let row = &logits.data[start..end];

        let probs = softmax(row);

        let target = targets[i];

        loss -= probs[target].ln();
    }

    loss / targets.len() as f32
}