mod tensor;
mod tokenizer;
mod nn;
mod transformer;
mod gpt;
mod inference;
mod training;
mod utils;

use crate::tokenizer::byte::ByteTokenizer;
use crate::gpt::model::GPT;
use crate::inference::generate::generate;
use crate::training::train::train;

fn main() {
    println!("=== Mini GPT (Rust) ===\n");

    // ---- 設定 ----
    let vocab_size = 256;   // ByteTokenizer は 0..255
    let n_layers = 2;       // Transformer ブロック数
    let dim = 64;           // 埋め込み次元
    let n_heads = 4;        // Attention head 数
    let seq_len = 16;       // 学習時のシーケンス長 (短いほうが学習しやすい)
    let epochs = 500;       // 学習エポック数
    let lr = 0.01;          // 学習率

    // ---- 学習データ ----
    // 3つの短い文の大量繰り返し
    let base = "the cat sat on the mat. the dog ran in the park. the sun is warm. ";
    let training_text: String = base.repeat(30);

    // ---- トークン化 ----
    let tokens = ByteTokenizer::encode(&training_text);
    println!("Training data: {} characters -> {} tokens", training_text.len(), tokens.len());
    println!("Model: vocab={}, layers={}, dim={}, heads={}", vocab_size, n_layers, dim, n_heads);
    println!("Training: epochs={}, lr={}, seq_len={}\n", epochs, lr, seq_len);

    // ---- モデル作成 ----
    let mut model = GPT::new(vocab_size, n_layers, dim, n_heads);

    // ---- 学習 ----
    println!("--- Training ---");
    train(&mut model, &tokens, epochs, lr, seq_len);

    // ---- 文章生成テスト ----
    println!("\n--- Generation ---");

    let prompts = vec![
        "the ",
        "the c",
        "the d",
        "the s",
        "the cat sat",
    ];

    for prompt in prompts {
        let mut input_tokens = ByteTokenizer::encode(prompt);
        generate(&model, &mut input_tokens, 60, 0.7, 5);
        let generated = ByteTokenizer::decode(&input_tokens);
        println!("Prompt: {:?}", prompt);
        println!("Output: {:?}\n", generated);
    }
}