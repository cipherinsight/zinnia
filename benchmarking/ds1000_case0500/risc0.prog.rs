// DS-1000 case 500 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 54;
    let mut a: [f64; N] = [0.0; N];
    for k in 0..N { a[k] = env::read(); }
    let mut claimed: [f64; N] = [0.0; N];
    for k in 0..N { claimed[k] = env::read(); }

    for i in 0..18 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let v = a[idx];
            assert!((v - claimed[idx]).abs() <= 0.001);
        }
    }
}
