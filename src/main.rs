mod tensor;
mod tokenizer;
mod nn;
mod transformer;
mod gpt;
mod inference;

use crate::gpt::model::GPT;
use crate::inference::generate::generate;
use crate::tokenizer::simple::SimpleTokenizer;

fn main() {

    let vocab = vec![
        "hello",
        "world",
        "rust",
        "gpt",
        "ai",
        "is",
        "cool",
        "!"
    ];

    let tokenizer = SimpleTokenizer::new(vocab);

    let model = GPT::new(8, 2, 32);

    let tokens = tokenizer.encode("hello");

    let output = generate(&model, tokens, 10);

    let text = tokenizer.decode(&output);

    println!("{}", text);
}