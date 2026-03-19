use crate::gpt::model::GPT;
use crate::tensor::softmax::softmax;
use crate::inference::sample::sample;
use crate::inference::topk::top_k;

/// Autoregressive text generation
/// tokens: 入力トークン列（インプレースで伸ばす）
/// max_new_tokens: 生成するトークン数
/// temperature: サンプリング温度（1.0=ニュートラル, <1でシャープ, >1でフラット）
/// top_k_val: top-k サンプリング（0なら無効）
pub fn generate(
    model: &GPT,
    tokens: &mut Vec<usize>,
    max_new_tokens: usize,
    temperature: f32,
    top_k_val: usize,
) {
    for _step in 0..max_new_tokens {
        let logits = model.forward(tokens);

        if logits.rows == 0 || logits.cols == 0 {
            eprintln!("generate: logits empty");
            break;
        }

        // 最後の行のlogitsを取得
        let start = (logits.rows - 1) * logits.cols;
        let last_row = &logits.data[start..start + logits.cols];

        // temperature scaling
        let mut probs_vec: Vec<f32> = last_row.to_vec();
        if temperature != 1.0 {
            for v in probs_vec.iter_mut() {
                *v /= temperature;
            }
        }

        // top-k filtering
        if top_k_val > 0 && top_k_val < probs_vec.len() {
            top_k(&mut probs_vec, top_k_val);
        }

        // softmax → sample
        let probs = softmax(&probs_vec);
        let next = sample(&probs);
        tokens.push(next);
    }
}