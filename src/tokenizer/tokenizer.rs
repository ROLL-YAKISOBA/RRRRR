use std::collections::HashMap;

pub struct Tokenizer {

    stoi: HashMap<char, usize>,
    itos: HashMap<usize, char>,

}


impl Tokenizer {

pub fn vocab_size(&self) -> usize {
    self.itos.len()
}

    pub fn decode(&self, tokens: &[usize]) -> String {

    tokens
        .iter()
        .map(|t| self.itos.get(t).unwrap())
        .collect()

}


    pub fn new(text: &str) -> Self {

        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();

        for (i, c) in chars.iter().enumerate() {

            stoi.insert(*c, i);
            itos.insert(i, *c);

        }

        Self { stoi, itos }

    }

    pub fn encode(&self, text: &str) -> Vec<usize> {

        text.chars()
            .map(|c| *self.stoi.get(&c).unwrap())
            .collect()

    }

}
