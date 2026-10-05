// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // Read input X[5][4]
    let mut x: [[i32; 4]; 5] = [[0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            x[i][j] = env::read();
        }
    }

    // Read result[5][4]
    let mut result: [[f32; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            result[i][j] = env::read();
        }
    }

    // Recompute expected
    let mut l1: [f32; 5] = [0.0; 5];
    for i in 0..5 {
        let mut s: f32 = 0.0;
        for j in 0..4 {
            let val = x[i][j] as f32;
            s += if val >= 0.0 { val } else { -val };
        }
        l1[i] = s;
    }

    for i in 0..5 {
        for j in 0..4 {
            let expected = (x[i][j] as f32) / l1[i];
            assert!((result[i][j] - expected).abs() < 1e-6);
        }
    }
}
