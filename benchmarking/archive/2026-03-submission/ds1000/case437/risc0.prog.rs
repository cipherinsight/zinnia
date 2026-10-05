// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input 3×2 matrix
    let mut a: [[i32; 2]; 3] = [[0; 2]; 3];
    for i in 0..3 {
        for j in 0..2 {
            a[i][j] = env::read();
        }
    }

    // Read mask
    let mut mask: [[i32; 2]; 3] = [[0; 2]; 3];
    for i in 0..3 {
        for j in 0..2 {
            mask[i][j] = env::read();
        }
    }

    // Compute expected mask (row-wise minima)
    for i in 0..3 {
        let mut row_min = a[i][0];
        for j in 1..2 {
            if a[i][j] < row_min {
                row_min = a[i][j];
            }
        }
        for j in 0..2 {
            let expected = if a[i][j] == row_min { 1 } else { 0 };
            assert_eq!(mask[i][j], expected);
        }
    }

    // env::commit(&output);
}
