// src/inference/generate.rs
use crate::gpt::model::GPT;
use crate::tensor::softmax::softmax;

use crate::inference::sample::sample;
use crate::inference::topk::top_k; 

/// simple generate: tokens をインプレースで伸ばす
/// temperature: 例 0.7, top_k: 例 40 (0なら top_k 無し)
pub fn generate(model: & GPT, tokens: &mut Vec<usize>, max_new_tokens: usize, temperature: f32, top_k_val: usize) {
    for _step in 0..max_new_tokens {
        let logits = model.forward(&tokens); // Tensor rows = seq, cols = vocab

        if logits.rows == 0 || logits.cols == 0 {
        
            eprintln!("generate: logits empty");
            break;
        }

      
        let start = (logits.rows - 1) * logits.cols;
        let last_row = &logits.data[start..start + logits.cols];

        // copy to Vec<f32> so we can modify (temperature, top_k)
        let mut probs_vec: Vec<f32> = last_row.to_vec();

       
        if top_k_val > 0 {
            top_k(&mut probs_vec, top_k_val);
        }

        let probs = softmax(&probs_vec);

        let next = sample(&probs);
        tokens.push(next);
    }
}


/* 

use crate::gpt::model::GPT;
use crate::tensor::tensor::Tensor;

pub fn generate(
    model: &mut GPT,
    tokens: &mut Vec<usize>,
    max_new_tokens: usize,
) {

    for _ in 0..max_new_tokens {

        // 入力
let logits = model.forward(&tokens);

        let next = argmax(&logits);

        tokens.push(next);

    }

}

fn argmax(t: &Tensor) -> usize {

    let mut max_i = 0;
    let mut max_v = t.data[0];

    for (i, v) in t.data.iter().enumerate() {

        if *v > max_v {

            max_v = *v;
            max_i = i;

        }

    }

    max_i
}



*/