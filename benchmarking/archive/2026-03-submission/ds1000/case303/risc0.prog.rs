// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input A
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..7 {
        a.push(env::read());
    }

    // read input B
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..2 {
            tmp.push(env::read());
        }
        b.push(tmp);
    }

    let ncol: usize = 2;
    let nrow: usize = 3;
    let truncated: Vec<i32> = vec![a[0], a[1], a[2], a[3], a[4], a[5]];

    for i in 0..nrow {
        for j in 0..ncol {
            let idx: usize = i * ncol + j;
            assert_eq!(b[i][j], truncated[idx]);
        }
    }

    // env::commit(&output);
}
