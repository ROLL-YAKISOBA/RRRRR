use crate::tensor::tensor::Tensor;


pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {


    //println!("MATMUL {:?} x {:?}", (a.rows,a.cols),(b.rows,b.cols));

    assert_eq!(a.cols, b.rows, "Matrix shape mismatch");

    let mut result = Tensor::new(a.rows, b.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {

            let mut sum = 0.0;

            for k in 0..a.cols {
                sum += a.data[i*a.cols + k] *
                       b.data[k*b.cols + j];
            }

            result.data[i*b.cols + j] = sum;
        }
    }

    result
}

pub fn add_backward(a: &mut Tensor, b: &mut Tensor, grad_out: &[f32]) {

    for i in 0..grad_out.len() {

        a.grad[i] += grad_out[i];
        b.grad[i] += grad_out[i];

    }

}

