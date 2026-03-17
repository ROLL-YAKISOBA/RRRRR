
use crate::tensor::tensor::Tensor;

pub fn positional_encoding(seq_len: usize, dim: usize) -> Tensor {

    let mut pe = Tensor::zeros(seq_len, dim);

    for pos in 0..seq_len {

        for i in 0..dim {

            let denom = 10000_f32.powf((2*(i/2)) as f32 / dim as f32);

            let value = pos as f32 / denom;

            if i % 2 == 0 {
                pe.data[pos*dim + i] = value.sin();
            } else {
                pe.data[pos*dim + i] = value.cos();
            }

        }

    }

    pe
}
