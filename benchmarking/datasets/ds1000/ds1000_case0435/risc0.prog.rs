// DS-1000 case 435 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { a.push(env::read()); }
    let mut claimed: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { claimed.push(env::read()); }

    for i in 0..4 {
        for j in 0..4 {
            let out: i64 = if i == 1 || j == 0 { 0 } else { a[i * 4 + j] };
            assert_eq!(out, claimed[i * 4 + j]);
        }
    }
}
