// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read input a (5x5)
    let mut a: Vec<Vec<i32>> = Vec::new();
    for _ in 0..5 {
        let mut row: Vec<i32> = Vec::new();
        for _ in 0..5 {
            row.push(env::read());
        }
        a.push(row);
    }

    // read result (5)
    let mut result: Vec<i32> = Vec::new();
    for _ in 0..5 {
        result.push(env::read());
    }

    // Step 1: fliplr
    let mut flipped: Vec<Vec<i32>> = vec![vec![0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            flipped[i as usize][j as usize] = a[i as usize][(4 - j) as usize];
        }
    }

    // Step 2: diagonal extraction
    let mut diag_vals: Vec<i32> = vec![0; 5];
    for k in 0..5 {
        diag_vals[k as usize] = flipped[k as usize][k as usize];
    }

    // Step 3: verify
    for k in 0..5 {
        assert_eq!(result[k as usize], diag_vals[k as usize]);
    }

    // env::commit(&output);
}
