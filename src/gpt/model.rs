use crate::nn::embedding::Embedding;
use crate::transformer::block::TransformerBlock;
use crate::nn::output::OutputLayer;
use crate::tensor::tensor::Tensor;


pub struct GPT {

    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub output: OutputLayer,

    pub vocab: usize,
    pub dim: usize,

}

impl GPT {

    pub fn new(vocab: usize, n_layer: usize, dim: usize) -> Self {

        let embedding = Embedding::new(vocab, dim);

        let mut blocks = Vec::with_capacity(n_layer);

        for _ in 0..n_layer {
            blocks.push(TransformerBlock::new(dim, 4));
        }

        let output = OutputLayer::new(dim, vocab);

        GPT {
            embedding,
            blocks,
            output,
            vocab,
            dim,
        }
    }

    pub fn encode(&self, tokens: &[usize]) -> crate::tensor::tensor::Tensor {
        
        let mut x = self.embedding.forward(tokens);

       
        let pos = crate::nn::position::positional_encoding(tokens.len(), self.dim);
        x = crate::tensor::tensor::Tensor::add(&x, &pos);

        for block in &self.blocks {
            x = block.forward(&x);
        }

        x
    }


    pub fn forward(&self, tokens: &[usize]) -> Tensor {

    let x = self.encode(tokens);

    let seq_len = x.rows;
    let dim = x.cols;

    let mut out_data = vec![0.0; seq_len * self.vocab];

    for i in 0..seq_len {

        let start = i * dim;
        let end = start + dim;

        let row = &x.data[start..end];

        let logits = self.output.forward(row);

        for v in 0..self.vocab {
            out_data[i * self.vocab + v] = logits[v];
        }
    }

    Tensor {
        data: out_data,
        rows: seq_len,
        cols: self.vocab,
        grad: vec![0.0; seq_len * self.vocab],
    }
}

}