// DS-1000 case 439 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut x: [[f64; 20]; 10] = [[0.0; 20]; 10];
    for i in 0..10 {
        for k in 0..20 {
            x[i][k] = env::read();
        }
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(20 * 10 * 10);
    for _ in 0..(20 * 10 * 10) {
        claimed.push(env::read());
    }

    let mut xt: [[f64; 10]; 20] = [[0.0; 10]; 20];
    for k in 0..20 {
        for i in 0..10 {
            xt[k][i] = x[i][k];
        }
    }

    let mut result: [[[f64; 10]; 10]; 20] = [[[0.0; 10]; 10]; 20];
    for k in 0..20 {
        for i in 0..10 {
            for j in 0..10 {
                result[k][i][j] = xt[k][i] * xt[k][j];
            }
        }
    }

    let mut idx = 0;
    for k in 0..20 {
        for i in 0..10 {
            for j in 0..10 {
                assert!((result[k][i][j] - claimed[idx]).abs() < 0.001);
                idx += 1;
            }
        }
    }
}
