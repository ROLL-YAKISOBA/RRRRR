mod tensor;
mod tokenizer;
mod nn;
mod transformer;
mod gpt;
mod inference;
mod training;

use crate::tokenizer::tokenizer::Tokenizer;
use crate::gpt::model::GPT;
use crate::inference::generate::generate;
use crate::training::train::train;

fn main() {

    let vocab_size = 256;

    let mut model = GPT::new(
        vocab_size,
        4,
        128
    );

    let text = "hello world this is test data";

    let tokenizer = Tokenizer::new(text);

    let mut tokens = tokenizer.encode("hello");

    train(&mut model,&tokens,10);

let out = generate(
    &model,
    &mut tokens,
    20,
    1.0,
    5
);

println!("{:?}", out);
    

  
}