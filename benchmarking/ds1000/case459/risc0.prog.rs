use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<f64> = Vec::new();
    for i in 0..4 {
        data.push(env::read());
    }
    let result: f64 = env::read();
    let answer = (data[0] + data[1] + data[2] + data[3]) / 4.0;
    assert_eq!(answer, result);

    // write public output to the journal
    // env::commit(&input);
}
