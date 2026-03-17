
use crate::tensor::tensor::Tensor;

pub fn causal_mask(size: usize) -> Tensor {

    let mut data = vec![0.0; size * size];

    for i in 0..size {
        for j in 0..size {

            if j > i {
                data[i*size + j] = -1e9;
            }

        }
    }

    Tensor {
        data,
        rows: size,
        cols: size,
        grad: vec![0.0; size * size],
    }
}