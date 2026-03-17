use crate::tensor::tensor::Tensor;

pub fn causal_mask(seq: usize) -> Tensor {

    let mut data = vec![0.0; seq * seq];

    for i in 0..seq {
        for j in 0..seq {

            if j > i {
                data[i*seq + j] = f32::NEG_INFINITY;
            }

        }
    }

    Tensor {
        data,
        rows: seq,
        cols: seq
    }
}