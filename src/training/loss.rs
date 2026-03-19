use crate::tensor::tensor::Tensor;
use crate::tensor::tensor::softmax;

pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    let mut loss = 0.0;
    let vocab = logits.cols;

    for i in 0..targets.len() {
        let start = i * vocab;
        let end = start + vocab;
        let row = &logits.data[start..end];

        let probs = softmax(row);
        let target = targets[i];

        // clamp to avoid log(0)
        let p = probs[target].max(1e-10);
        loss -= p.ln();
    }

    loss / targets.len() as f32
}