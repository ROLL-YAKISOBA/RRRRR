use crate::tensor::tensor::Tensor;

pub struct Adam {

    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,

    pub m: Vec<f32>,
    pub v: Vec<f32>,

}

impl Adam {

    pub fn new(size: usize, lr: f32) -> Self {

        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            m: vec![0.0; size],
            v: vec![0.0; size],
        }

    }

    pub fn step(&mut self, param: &mut Tensor) {

        for i in 0..param.data.len() {

            let g = param.grad[i];

            self.m[i] = self.beta1*self.m[i] + (1.0-self.beta1)*g;
            self.v[i] = self.beta2*self.v[i] + (1.0-self.beta2)*g*g;

            param.data[i] -= self.lr * self.m[i] / (self.v[i].sqrt()+1e-8);

        }

    }

}