//use crate::tensor::Tensor;
use rand;
use crate::nn::position::positional_encoding;
use crate::tensor::tensor::Tensor;
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {


    
    pub fn new(vocab_size: usize, dim: usize) -> Self {

        let mut data = vec![0.0; vocab_size * dim];

        for i in 0..data.len() {
            data[i] = rand::random::<f32>() * 0.01;
        }

        Self {
    weight: Tensor {
        data,
        rows: vocab_size,
        cols: dim,
        grad: vec![0.0; vocab_size * dim],
    }
}
        /* 
        Self {
            // ✅ 修正：vocab_size と dim を使う
            weight: Tensor::new(vocab_size, dim),
        }*/
    }

pub fn forward(&self, tokens: &[usize]) -> Tensor {

    let seq_len = tokens.len();
    let dim = self.weight.cols;

    let mut out = Vec::with_capacity(seq_len * dim);

    for &t in tokens {

        let start = t * dim;
        let end = start + dim;

        out.extend_from_slice(&self.weight.data[start..end]);

    }

    let token_embedding = Tensor {
        data: out,
        rows: seq_len,
        cols: dim,
        grad: vec![0.0; seq_len * dim],
    };

    let pos = positional_encoding(seq_len, dim);

    Tensor::add(&token_embedding, &pos)

}

}

