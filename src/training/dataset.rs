pub fn build_dataset(tokens: &Vec<usize>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..tokens.len()-1 {

        inputs.push(tokens[..=i].to_vec());
        targets.push(tokens[1..=i+1].to_vec());

    }

    (inputs, targets)
}