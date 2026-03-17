
 use rand::Rng; 

pub fn top_k_sample(probs: &[f32], k: usize) -> usize {

    let mut pairs: Vec<(usize,f32)> =
        probs.iter().enumerate().map(|(i,p)|(i,*p)).collect();

    pairs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

    let top = &pairs[..k];

    let sum: f32 = top.iter().map(|(_,p)| *p).sum();

    let mut rng = rand::thread_rng();

    let mut r = rng.gen::<f32>() * sum;

    for (i,p) in top {

        r -= *p;

        if r <= 0.0 {
            return *i;
        }

    }

    top[0].0
}


/*

pub fn top_k_sample(probs: &Vec<f32>, k: usize) -> usize {

    let mut pairs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top = &pairs[..k];

    let mut sum = 0.0;

    for (_, p) in top {
        sum += p;
    }

    let mut rng = rand::thread_rng();
    let mut r = rng.gen::<f32>() * sum;

    for (i, p) in top {

        r -= p;

        if r <= 0.0 {
            return *i;
        }

    }

    top[0].0
}
*/