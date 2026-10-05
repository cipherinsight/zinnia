use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let power: f64 = env::read();
    let mut answers: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(env::read());
        }
        answers.push(tmp);
    }

    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(data[i][j].powf(power),  answers[i][j]);
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
