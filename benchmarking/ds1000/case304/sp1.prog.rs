// SP1 program code
#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    // read input A (7)
    let mut a: Vec<i32> = Vec::new();
    for _ in 0..7 {
        a.push(sp1_zkvm::io::read::<i32>());
    }

    // read input B (3x2)
    let mut b: Vec<Vec<i32>> = Vec::new();
    for _ in 0..3 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(sp1_zkvm::io::read::<i32>());
        }
        b.push(row);
    }

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

    // reshaped = reversed_part.reshape((3,2))
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

    // sp1_zkvm::io::commit_slice(&output);
}
