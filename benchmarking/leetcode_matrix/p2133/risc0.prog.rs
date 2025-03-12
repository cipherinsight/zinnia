use risc0_zkvm::guest::env;

fn main() {
    // TODO: Implement your guest code here
    // read the input
    let mut matrix: Vec<Vec<i32>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(env::read());
        }
        matrix.push(tmp);
    }
    let valid = env::read();

    assert!(
        (0..=1).contains(&valid),
        "Valid must be either 0 or 1, but got {}",
        valid
    );

    let n = matrix.len();

    for row in matrix.iter() {
        for &x in row.iter() {
            assert!(
                valid == 0 || (1 < x && x <= n as i32),
                "Invalid value {} in matrix when valid = {}",
                x,
                valid
            );
        }
    }

    // write public output to the journal
    // env::commit(&input);
}
