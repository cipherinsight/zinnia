// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read the input
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..3 {
        a.push(env::read());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..4 {
            tmp.push(env::read());
        }
        result.push(tmp);
    }

    for i in 0..3 {
        for j in 0..4 {
            let expected: i32 = if a[i] == j { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }

    // env::commit(&output);
}
