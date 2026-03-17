use crate::gpt::model::GPT;
use crate::training::loss::cross_entropy;

pub fn train(model: &mut GPT, data: &Vec<usize>, epochs: usize) {

    for e in 0..epochs {

        let logits = model.forward(data);

        let loss = cross_entropy(&logits, data);

        println!("epoch {} loss {}", e, loss);

        // TODO backprop

    }

}