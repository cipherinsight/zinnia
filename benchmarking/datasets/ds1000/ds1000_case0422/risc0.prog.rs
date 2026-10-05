// DS-1000 case 422 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { a.push(env::read()); }
    let mut b: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { b.push(env::read()); }
    let mut claimed: Vec<f64> = Vec::with_capacity(50);
    for _ in 0..50 { claimed.push(env::read()); }

    let mut c: Vec<f64> = Vec::with_capacity(99);
    for i in 1..50 { c.push(b[i]); }
    for i in 0..50 { c.push(b[i]); }

    for k in 0..50 {
        let mut s: f64 = 0.0;
        for n in 0..50 {
            s += a[n] * c[n + 49 - k];
        }
        assert!((s - claimed[k]).abs() < 0.001);
    }
}
