// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input 4x4 matrix
    let mut a: [[i32; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = env::read();
        }
    }

    // Read expected result
    let mut result: [[i32; 4]; 4] = [[0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = env::read();
        }
    }

    let zero_rows: [usize; 2] = [1, 3];
    let zero_cols: [usize; 2] = [1, 2];

    let mut modified = a;

    for &r in zero_rows.iter() {
        for j in 0..4 {
            modified[r][j] = 0;
        }
    }
    for &c in zero_cols.iter() {
        for i in 0..4 {
            modified[i][c] = 0;
        }
    }

    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(result[i][j], modified[i][j]);
        }
    }

    // env::commit(&output);
}
