pub fn sgd(weights: &mut Vec<f32>, grads: &Vec<f32>, lr: f32) {

    for i in 0..weights.len() {
        weights[i] -= lr * grads[i];
    }

}