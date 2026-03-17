pub fn cross_entropy(probs: &Vec<f32>, target: usize) -> f32 {

    let p = probs[target].max(1e-9);

    -p.ln()

}