use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..5 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let mut results: Vec<f64> = Vec::new();
    for i in 0..2 {
        results.push(env::read());
    }

    let mut answers = vec![0.0; 2];
    for i in 0..2 {
        let mut sum = 0.0;
        let bins = 5 / 3;
        for j in (5%3)..5 {
            sum += data[i][j];
        }
        answers[i] = sum / (bins as f64) / 3.0;
    }

    assert_eq!(results[0], answers[0]);
    assert_eq!(results[1], answers[1]);

    // write public output to the journal
    // env::commit(&input);
}
