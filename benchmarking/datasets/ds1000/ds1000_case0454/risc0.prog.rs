// DS-1000 case 454 — Risc0 guest. Linf row-normalisation.

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
        let mut linf: f64 = abs_f[i][0];
        for j in 1..4 {
            if abs_f[i][j] > linf {
                linf = abs_f[i][j];
            }
        }
        assert!(linf != 0.0);
        for j in 0..4 {
            result[i][j] = x_f[i][j] / linf;
        }
    }

    for i in 0..5 {
        for j in 0..4 {
            assert!((result[i][j] - claimed[i][j]).abs() < 0.001);
        }
    }
}
