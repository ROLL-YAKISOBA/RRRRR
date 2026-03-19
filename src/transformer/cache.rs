use crate::tensor::tensor::Tensor;

pub struct KVCache {

    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,

}

impl KVCache {

    pub fn new() -> Self {

        Self {
            k: Vec::new(),
            v: Vec::new(),
        }

    }

}