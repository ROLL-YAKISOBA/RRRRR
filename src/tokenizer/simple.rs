use std::collections::HashMap;

pub struct SimpleTokenizer {

    pub stoi: HashMap<String, usize>,
    pub itos: HashMap<usize, String>

}

impl SimpleTokenizer {

    pub fn new(words: Vec<&str>) -> Self {

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();

        for (i, w) in words.iter().enumerate() {
            stoi.insert(w.to_string(), i);
            itos.insert(i, w.to_string());
        }

        Self { stoi, itos }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {

        text.split_whitespace()
            .map(|w| *self.stoi.get(w).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, tokens: &Vec<usize>) -> String {

        tokens.iter()
            .map(|t| self.itos.get(t).unwrap())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}