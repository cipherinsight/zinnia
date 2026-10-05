// DS-1000 case 336 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            a[i][j] = env::read();
        }
    }
    let mut result = [0.0f64; 8];
    for i in 0..8 { result[i] = env::read(); }

    for i in 0..8 {
        let computed = a[i][7 - i];
        assert!((result[i] - computed).abs() < 1e-12);
    }
}
