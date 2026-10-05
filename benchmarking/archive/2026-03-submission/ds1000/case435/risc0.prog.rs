// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input 4×4 matrix
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

    let row = 1usize;
    let col = 0usize;

    let mut modified = a;
    for j in 0..4 {
        modified[row][j] = 0;
    }
    for i in 0..4 {
        modified[i][col] = 0;
    }

    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(result[i][j], modified[i][j]);
        }
    }

    // env::commit(&output);
}
