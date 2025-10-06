// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input a (2x2)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read());
        }
        a.push(row);
    }

    // read input result (2x2)
    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..2 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..2 {
            row.push(env::read());
        }
        result.push(row);
    }

    // Step 1: Compute min value
    let mut min_val: i32 = a[0][0];
    for i in 0..2 {
        for j in 0..2 {
            if a[i as usize][j as usize] < min_val {
                min_val = a[i as usize][j as usize];
            }
        }
    }

    // Step 2: Build expected matrix
    let mut expected: Vec<Vec<i32>> = vec![vec![0; 2], vec![0; 2]];
    let mut idx: usize = 0;
    for i in 0..2 {
        for j in 0..2 {
            if a[i as usize][j as usize] == min_val {
                expected[idx]
