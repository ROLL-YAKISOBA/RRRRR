pub struct OutputLayer {

    pub weight: Vec<f32>,
    pub dim: usize,
    pub vocab: usize,

}

impl OutputLayer {

    pub fn new(dim: usize, vocab: usize) -> Self {

        Self {

            weight: vec![0.01; dim * vocab],
            dim,
            vocab,

        }

    }

}



impl OutputLayer {

    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {

        let mut logits = vec![0.0; self.vocab];

        for v in 0..self.vocab {

            let mut sum = 0.0;

            for d in 0..self.dim {

                sum += x[d] * self.weight[d*self.vocab + v];

            }

            logits[v] = sum;

        }

        logits

    }

}