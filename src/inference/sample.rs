use rand::Rng;

pub fn sample(prob: &Vec<f32>) -> usize {

    let mut rng = rand::thread_rng();

    let mut cumulative = 0.0;

    let r: f32 = rng.gen();

    for (i,p) in prob.iter().enumerate() {

        cumulative += p;

        if r < cumulative {
            return i;
        }

    }

    prob.len()-1

}