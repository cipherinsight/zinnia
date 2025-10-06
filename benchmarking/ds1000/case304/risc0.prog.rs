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

    // Zinnia logic: truncated = A[1:], reversed_part = truncated[::-1], reshaped = reversed_part.reshape((3,2))
    let mut truncated: Vec<i32> = Vec::new();
    for i in 1..7 {
        truncated.push(a[i]);
    }

    let mut reversed_part: Vec<i32> = Vec::new();
    for i in (0..truncated.len()).rev() {
        reversed_part.push(truncated[i]);
    }

    // reshape reversed_part (len = 6) into (3,2)
    let mut reshaped: Vec<Vec<i32>> = Vec::new();
    let nrow: usize = 3;
    let ncol: usize = 2;
    for i in 0..nrow {
        let mut row: Vec<i32> = Vec::new();
        for j in 0..ncol {
            let idx: usize = i * ncol + j;
            row.push(reversed_part[idx]);
        }
        re
