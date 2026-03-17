pub fn top_k(logits: &mut Vec<f32>, k: usize) {

    let mut pairs: Vec<(usize,f32)> =
        logits.iter().cloned().enumerate().collect();

    pairs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

    let threshold = pairs[k].1;

    for v in logits.iter_mut() {

        if *v < threshold {
            *v = -1e9;
        }

    }

}