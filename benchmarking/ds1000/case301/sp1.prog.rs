// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read inputs
    let mut A: Vec<i32> = Vec::new();
    for _ in 0..6 {
        A.push(sp1_zkvm::io::read::<i32>());
    }

    let mut B: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..2 {
            tmp.push(sp1_zkvm::io::read::<i32>());
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

    // sp1_zkvm::io::commit_slice(&output);
}
