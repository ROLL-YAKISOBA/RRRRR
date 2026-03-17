use crate::gpt::model::GPT;
use crate::tensor::tensor::Tensor;
use crate::inference::sample::sample;

pub fn generate(model: &GPT, mut tokens: Vec<usize>, steps: usize) -> Vec<usize> {

    for _ in 0..steps {

        let logits = model.forward(&tokens);

        let last_row = logits.rows - 1;

        let start = last_row * logits.cols;
        let end = start + logits.cols;

        let probs = Tensor {
            data: logits.data[start..end].to_vec(),
            rows: 1,
            cols: logits.cols
        };

        let next = sample(&probs);

        tokens.push(next);
    }

    tokens
}