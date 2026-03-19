use crate::tensor::tensor::Tensor;

pub fn positional_encoding(seq_len: usize, dim: usize) -> Tensor {
    let mut data = vec![0.0; seq_len * dim];
    for pos in 0..seq_len {
        for i in 0..dim {
            let angle = pos as f32 / 10000f32.powf((2 * (i / 2)) as f32 / dim as f32);
            if i % 2 == 0 {
                data[pos * dim + i] = angle.sin();
            } else {
                data[pos * dim + i] = angle.cos();
            }
        }
    }
    Tensor {
        data,
        rows: seq_len,
        cols: dim,
        grad: vec![0.0; seq_len * dim],
    }
}