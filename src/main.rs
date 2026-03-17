mod tensor;
mod tokenizer;
mod nn;
mod transformer;
mod gpt;
mod inference;
mod training;

use crate::gpt::model::GPT;
use crate::inference::generate::generate;
use crate::tokenizer::simple::SimpleTokenizer;

use crate::training::train::train;

fn main() {

    let vocab = vec![
        "hello","world","rust","gpt","ai","is","cool","!"
    ];

    let tokenizer = SimpleTokenizer::new(vocab);

    let mut model = GPT::new(8,2,32);

    let tokens = tokenizer.encode("hello world hello world");

    train(&mut model,&tokens,10);

}