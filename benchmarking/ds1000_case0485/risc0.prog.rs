// DS-1000 case 485 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[[f64; 3]; 6]; 5] = [[[0.0; 3]; 6]; 5];
    for i in 0..5 {
        for j in 0..6 {
            for p in 0..3 {
                a[i][j][p] = env::read();
            }
        }
    }
    let mut b: [[f64; 3]; 3] = [[0.0; 3]; 3];
    for p in 0..3 {
        for k in 0..3 {
            b[p][k] = env::read();
        }
    }
    let mut claimed: [f64; 90] = [0.0; 90];
    for idx in 0..90 {
        claimed[idx] = env::read();
    }

    for i in 0..5 {
        for j in 0..6 {
            for k in 0..3 {
                let mut acc: f64 = 0.0;
                for p in 0..3 {
                    acc += a[i][j][p] * b[p][k];
                }
                let idx = i * 18 + j * 3 + k;
                assert!((acc - claimed[idx]).abs() <= 0.005);
            }
        }
    }
}
