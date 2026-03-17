use crate::gpt::model::GPT;
use crate::training::loss::cross_entropy_loss;


//use crate::nn::output::OutputLayer;
use crate::tensor::tensor::Tensor;
use crate::tensor::softmax::softmax; // slice softmax




pub fn train(model: &mut GPT, data: &Vec<usize>, epochs: usize) {

    for e in 0..epochs {

        let logits = model.forward(data);

        let loss = cross_entropy_loss(&logits, data);

        println!("epoch {} loss {}", e, loss);

        // TODO backprop

    }

}



/// 出力層のみを1ステップ更新する（バッチsize=1 の簡易版）
pub fn train_output_layer_step(model: &mut GPT, input: &[usize], target: &[usize], lr: f32) {
    // 1) 隠れ表現（seq x dim）を取得
    let hidden = model.encode(input); // Tensor (seq x dim)

    // 2) 出力層への一括 forward（seq x vocab）
    let logits = model.output.forward_tensor(&hidden); // Tensor (seq x vocab)

    let seq = logits.rows;
    let vocab = logits.cols;
    assert_eq!(seq, target.len());

    // 3) 行ごとに softmax -> grad_logits = p - one_hot(target)
    // 勾配テンソルを作る: (seq x vocab)
    let mut grad_data = vec![0.0; seq * vocab];

    for i in 0..seq {
        let start = i * vocab;
        let row = &logits.data[start..start+vocab];
        let probs = softmax(row); // returns Vec<f32> length=vocab

        let t = target[i];
        for v in 0..vocab {
            let mut g = probs[v];
            if v == t { g -= 1.0; } // grad of cross-entropy w.r.t logits = p - 1_{v=target}
            grad_data[start + v] = g;
        }
    }

    let grad_logits = Tensor {
        data: grad_data,
        rows: seq,
        cols: vocab,
        grad: vec![0.0; seq * vocab],
    };

    // 4) 出力層の重みを更新（batch 合成 dW）: dW = hidden^T * grad_logits  (we update inside)
    model.output.update_from_grad_batch(&hidden, &grad_logits, lr);

    // 5) (任意) loss を計算して表示
    let loss = cross_entropy_loss(&logits, target);
    println!("train step loss = {:.6}", loss);
}