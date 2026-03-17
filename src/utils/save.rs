use std::fs::File;
use std::io::Write;
use crate::tensor::tensor::Tensor;

pub fn save_tensor(path: &str, t: &Tensor) {

    let mut f = File::create(path).unwrap();

    for v in &t.data {

        writeln!(f,"{}",v).unwrap();

    }

}