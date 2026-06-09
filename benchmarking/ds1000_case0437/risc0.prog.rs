// DS-1000 case 437 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(env::read()); }
    let mut claimed: Vec<u32> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(env::read()); }

    for i in 0..10 {
        let mut row_min = a[i * 5];
        for j in 1..5 {
            let cur = a[i * 5 + j];
            if cur < row_min { row_min = cur; }
        }
        for j in 0..5 {
            let diff = a[i * 5 + j] - row_min;
            let eq: u32 = if diff.abs() < 0.001 { 1 } else { 0 };
            assert_eq!(eq, claimed[i * 5 + j]);
        }
    }
}
