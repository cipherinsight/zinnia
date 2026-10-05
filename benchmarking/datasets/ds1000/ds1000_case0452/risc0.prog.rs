// DS-1000 case 452 — Risc0 guest. L1 row-normalisation on Integer[5,4].

use risc0_zkvm::guest::env;

fn main() {
    let mut x: [[i64; 4]; 5] = [[0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            x[i][j] = env::read();
        }
    }
    let mut claimed: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            claimed[i][j] = env::read();
        }
    }

    let mut x_f: [[f64; 4]; 5] = [[0.0; 4]; 5];
    let mut abs_f: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            let v = x[i][j] as f64;
            x_f[i][j] = v;
            abs_f[i][j] = v.abs();
        }
    }

    let mut result: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        let mut l1: f64 = 0.0;
        for j in 0..4 {
            l1 = l1 + abs_f[i][j];
        }
        assert!(l1 != 0.0);
        for j in 0..4 {
            result[i][j] = x_f[i][j] / l1;
        }
    }

    for i in 0..5 {
        for j in 0..4 {
            assert!((result[i][j] - claimed[i][j]).abs() < 0.001);
        }
    }
}
