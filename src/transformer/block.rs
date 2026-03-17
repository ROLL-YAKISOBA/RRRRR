use crate::transformer::multihead::MultiHeadAttention;
use crate::nn::feedforward::FeedForward;
use crate::tensor::tensor::Tensor;
use crate::transformer::layernorm::LayerNorm;

pub struct TransformerBlock {

    pub attn: MultiHeadAttention,
    pub ffn: FeedForward,

    pub norm1: LayerNorm,
    pub norm2: LayerNorm,

}

impl TransformerBlock {

    pub fn new(d: usize, hidden: usize) -> Self {

Self {

    attn: MultiHeadAttention::new(d, 4),
    ffn: FeedForward::new(d, hidden),

    norm1: LayerNorm::new(d),
    norm2: LayerNorm::new(d),

}

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

let x1 = self.norm1.forward(x);

let attn = self.attn.forward(&x1);

let x2 = Tensor::add(x, &attn);

let x3 = self.norm2.forward(&x2);

let ff = self.ffn.forward(&x3);



        Tensor::add(&x2, &ff)
    }
}




/*
use crate::tensor::tensor::Tensor;
use crate::transformer::multihead::MultiHeadAttention;
use crate::nn::feedforward::FeedForward;
use crate::transformer::layernorm::LayerNorm;

pub struct TransformerBlock {

    attn: MultiHeadAttention,
    ff: FeedForward,

    norm1: LayerNorm,
    norm2: LayerNorm,

}

impl TransformerBlock {

    pub fn new(dim: usize) -> Self {

        Self {

            attn: MultiHeadAttention::new(dim, 4),
            ff: FeedForward::new(dim),

            norm1: LayerNorm::new(dim),
            norm2: LayerNorm::new(dim),

        }

    }

    pub fn forward(&self, x: &Tensor) -> Tensor {

        // ---- Attention ----
        let norm_x = self.norm1.forward(x);
        let attn = self.attn.forward(&norm_x);

        let x1 = Tensor::add(x, &attn);

        // ---- FeedForward ----
        let norm_x1 = self.norm2.forward(&x1);
        let ff = self.ff.forward(&norm_x1);

        Tensor::add(&x1, &ff)

    }

}

*/