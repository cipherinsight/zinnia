// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read inputs
    let mut A: Vec<i32> = Vec::new();
    for _ in 0..6 {
        A.push(env::read());
    }

    let mut B: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..2 {
            tmp.push(env::read());
        }
        B.push(tmp);
    }

    let nrow: usize = 3;
    let ncol: usize = 2;

    for i in 0..nrow {
        for j in 0..ncol {
            let idx = i * ncol + j;
            assert_eq!(B[i][j], A[idx]);
        }
    }

    // env::commit(&output);
}
