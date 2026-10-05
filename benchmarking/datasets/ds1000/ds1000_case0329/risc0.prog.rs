// DS-1000 case 329 — Risc0 guest. Elementwise a ** 3 on Float[8, 6].

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [[0.0f64; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = env::read();
        }
    }
    let mut result = [[0.0f64; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            result[i][j] = env::read();
        }
    }

    for i in 0..8 {
        for j in 0..6 {
            let computed = a[i][j].powi(3);
            assert!((computed - result[i][j]).abs() < 1e-6);
        }
    }
}
