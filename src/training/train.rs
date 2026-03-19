use crate::gpt::model::GPT;
use crate::training::loss::cross_entropy_loss;
use crate::tensor::tensor::{Tensor, matmul};
use crate::tensor::softmax::softmax;

/// 学習データからスライディングウィンドウで走査し、
/// 出力層とエンベディングの重みを更新する訓練ループ
pub fn train(model: &mut GPT, data: &[usize], epochs: usize, lr: f32, seq_len: usize) {
    if data.len() < 2 {
        println!("Training data too short");
        return;
    }

    let effective_seq = seq_len.min(data.len() - 1);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        let mut pos = 0;
        while pos + effective_seq < data.len() {
            let input = &data[pos..pos + effective_seq];
            let target = &data[pos + 1..pos + effective_seq + 1];

            let loss = train_step(model, input, target, lr);
            total_loss += loss;
            n_batches += 1;

            pos += effective_seq;
        }

        let avg_loss = if n_batches > 0 {
            total_loss / n_batches as f32
        } else {
            0.0
        };
        if (epoch + 1) % 20 == 0 || epoch == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{} | avg loss: {:.4}", epoch + 1, epochs, avg_loss);
        }
    }
}

/// 勾配クリッピング
fn clip_grad(v: &mut [f32], max_val: f32) {
    for g in v.iter_mut() {
        if *g > max_val { *g = max_val; }
        if *g < -max_val { *g = -max_val; }
    }
}

/// 1ステップの訓練: forward → loss → 出力層 + エンベディングの勾配更新
fn train_step(model: &mut GPT, input: &[usize], target: &[usize], lr: f32) -> f32 {
    let grad_clip = 1.0;

    // ======== Forward Pass ========
    let hidden = model.encode(input); // (seq x dim)
    let logits = model.output.forward_tensor(&hidden); // (seq x vocab)

    let seq = logits.rows;
    let vocab = logits.cols;
    let dim = model.dim;

    // ======== Loss ========
    let loss = cross_entropy_loss(&logits, target);

    // ======== Backward: dL/d_logits ========
    let mut grad_logits_data = vec![0.0; seq * vocab];
    for i in 0..seq {
        let start = i * vocab;
        let row = &logits.data[start..start + vocab];
        let probs = softmax(row);
        let t = target[i];
        for v in 0..vocab {
            let mut g = probs[v];
            if v == t { g -= 1.0; }
            grad_logits_data[start + v] = g / seq as f32;
        }
    }
    clip_grad(&mut grad_logits_data, grad_clip);

    let grad_logits = Tensor {
        data: grad_logits_data,
        rows: seq,
        cols: vocab,
        grad: vec![],
    };

    // ======== 出力層の重み更新 ========
    model.output.update_from_grad_batch(&hidden, &grad_logits, lr);

    // ======== エンベディングの更新 ========
    // grad_hidden = grad_logits * W_output^T
    let w_out_t = model.output.weight.transpose();
    let mut grad_hidden = matmul(&grad_logits, &w_out_t);
    clip_grad(&mut grad_hidden.data, grad_clip);

    for i in 0..seq {
        let token = input[i];
        let grad_start = i * dim;
        let emb_start = token * dim;

        for d in 0..dim {
            let g = grad_hidden.data[grad_start + d];
            model.embedding.weight.data[emb_start + d] -= lr * g;
        }
    }

    loss
}