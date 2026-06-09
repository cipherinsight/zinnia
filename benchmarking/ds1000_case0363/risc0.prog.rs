// DS-1000 case 363 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a = [0.0f64; 100];
    for i in 0..100 { a[i] = env::read(); }
    let pos: i64 = env::read();
    let element: i64 = env::read();
    let mut claimed = [0.0f64; 101];
    for i in 0..101 { claimed[i] = env::read(); }

    let pos_u: usize = pos as usize;
    let elem_f: f64 = element as f64;
    let mut result = [0.0f64; 101];
    for i in 0..101 {
        if i < pos_u {
            result[i] = a[i];
        } else if i == pos_u {
            result[i] = elem_f;
        } else {
            result[i] = a[i - 1];
        }
    }
    for i in 0..101 {
        assert!((result[i] - claimed[i]).abs() < 1e-3);
    }
}
