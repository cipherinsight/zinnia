// DS-1000 case 334 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [0.0f64; 50];
    let mut b = [0.0f64; 50];
    let mut c = [0.0f64; 50];
    let mut result = [0.0f64; 50];
    for i in 0..50 { a[i] = env::read(); }
    for i in 0..50 { b[i] = env::read(); }
    for i in 0..50 { c[i] = env::read(); }
    for i in 0..50 { result[i] = env::read(); }

    for i in 0..50 {
        let computed = (a[i] + b[i] + c[i]) / 3.0;
        assert!((computed - result[i]).abs() < 1e-9);
    }
}
