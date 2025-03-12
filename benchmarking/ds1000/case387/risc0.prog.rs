use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut data: Vec<Vec<u64>> = Vec::new();
    for i in 0..4 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<u64>> = Vec::new();
    for i in 0..4 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(env::read());
        }
        results.push(tmp);
    }

    assert_eq!(data[0][0], results[0][0]);
    assert_eq!(data[0][1], results[0][1]);
    assert_eq!(data[0][2], results[1][0]);
    assert_eq!(data[0][3], results[1][1]);
    assert_eq!(data[1][0], results[0][2]);
    assert_eq!(data[1][1], results[0][3]);
    assert_eq!(data[1][2], results[1][2]);
    assert_eq!(data[1][3], results[1][3]);
    assert_eq!(data[2][0], results[2][0]);
    assert_eq!(data[2][1], results[2][1]);
    assert_eq!(data[2][2], results[3][0]);
    assert_eq!(data[2][3], results[3][1]);
    assert_eq!(data[3][0], results[2][2]);
    assert_eq!(data[3][1], results[2][3]);
    assert_eq!(data[3][2], results[3][2]);
    assert_eq!(data[3][3], results[3][3]);

    // write public output to the journal
    // env::commit(&input);
}
