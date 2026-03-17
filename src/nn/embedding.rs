use crate::tensor::Tensor;
use rand;

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
    }
}
        /* 
        Self {
            // ✅ 修正：vocab_size と dim を使う
            weight: Tensor::new(vocab_size, dim),
        }*/
    }

pub fn forward(&self, tokens: &[usize]) -> Tensor {

    let dim = self.weight.cols;
    let mut out = vec![];

    for &t in tokens {

        let id = if t < self.weight.rows { t } else { 0 };

        let start = id * dim;
        let end = start + dim;

        out.extend_from_slice(&self.weight.data[start..end]);
    }

    let mut tensor = Tensor {
        data: out,
        rows: tokens.len(),
        cols: dim
    };

    let pos = crate::nn::position::positional_encoding(tokens.len(), dim);

    for i in 0..tensor.data.len() {
        tensor.data[i] += pos.data[i];
    }

    tensor
}

}

