use crate::tensor::tensor::Tensor;

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    crate::tensor::tensor::matmul(a, b)
}

pub fn add_backward(a: &mut Tensor, b: &mut Tensor, grad_out: &[f32]) {
    for i in 0..grad_out.len() {
        a.grad[i] += grad_out[i];
        b.grad[i] += grad_out[i];
    }
}
