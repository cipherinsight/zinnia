use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<u64>> = Vec::new();
    for i in 0..(3 - 1) {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        results.push(tmp);
    }

    for i in 0..(3 - 1) {
        for j in 0..4 {
            assert_eq!(data[i][j], results[i][j]);
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
