// DS-1000 case 335 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(env::read()); }
    let mut b: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { b.push(env::read()); }
    let mut c: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { c.push(env::read()); }
    let mut claimed: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(env::read()); }

    for k in 0..50usize {
        let ab = if a[k] > b[k] { a[k] } else { b[k] };
        let m = if ab > c[k] { ab } else { c[k] };
        assert!((m - claimed[k]).abs() < 1e-9);
    }
}
