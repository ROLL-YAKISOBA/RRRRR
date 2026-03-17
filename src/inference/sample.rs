//use crate::tensor::tensor::Tensor;
//use rand::Rng;

use crate::tensor::tensor::Tensor;
use rand::Rng;

pub fn sample(probs: &Tensor) -> usize {

    let mut rng = rand::thread_rng();

    let mut cumulative = 0.0;
    let r: f32 = rng.gen();

    for (i, p) in probs.data.iter().enumerate() {

        cumulative += p;

        if r < cumulative {
            return i;
        }
    }

    probs.data.len() - 1
}