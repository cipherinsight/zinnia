// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input A (7)
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..7 {
        a.push(env::read());
    }

    // read input B (3x2)
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read());
        }
        b.push(row);
    }

    // Zinnia logic:
    // truncated = A[1:]
    let mut truncated: Vec<i32> = Vec::new();
    for i in 1..7 {
        truncated.push(a[i as usize]);
    }

    // reversed_part = truncated[::-1]
    let mut reversed_part: Vec<i32> = Vec::new();
    for i in (0..truncated.len()).rev() {
        reversed_part.push(truncated[i as usize]);
    }

    // reshaped = reversed_part.reshape((3,2))  (row-major)
    let nrow: usize = 3;
    let ncol: usize = 2;
    let mut reshaped: Vec<Vec<i32>> = Vec::new();
    for i in 0..nrow {
        let mut row: Vec<i32> = Vec::new();
        for j in 0..ncol {
            let idx: usize = i * ncol + j;
            row.push(reversed_part[idx]);
        }
        reshaped.push(row);
    }

    // assert B == reshaped
    for i in 0..nrow {
        for j in 0..ncol {
            assert_eq!(b[i as usize][j as usize], reshaped[i as usize][j as usize]);
        }
    }

    // env::commit(&output);
}
