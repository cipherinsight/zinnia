// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read a
    let mut a: [i32; 3] = [0; 3];
    for i in 0..3 {
        a[i] = env::read();
    }

    // Read result
    let mut result: [[i32; 4]; 3] = [[0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            result[i][j] = env::read();
        }
    }

    // Verify
    for i in 0..3 {
        for j in 0..4 {
            let expected = if a[i] == j as i32 { 1 } else { 0 };
            assert_eq!(result[i][j], expected);
        }
    }
}
