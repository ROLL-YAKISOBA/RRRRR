pub struct ByteTokenizer;

impl ByteTokenizer {

    pub fn encode(text: &str) -> Vec<usize> {

        text.bytes().map(|b| b as usize).collect()

    }

    pub fn decode(tokens: &[usize]) -> String {

        tokens.iter().map(|&t| t as u8 as char).collect()

    }

}