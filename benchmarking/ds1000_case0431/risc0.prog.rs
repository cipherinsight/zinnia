// DS-1000 case 431 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a_int: Vec<i64> = Vec::with_capacity(8);
    for _ in 0..8 { a_int.push(env::read()); }
    let mut claimed: Vec<u32> = Vec::with_capacity(8);
    for _ in 0..8 { claimed.push(env::read()); }

    let a: Vec<f64> = a_int.iter().map(|v| *v as f64).collect();

    let mut sum: f64 = 0.0;
    for v in a.iter() { sum += *v; }
    let mean = sum / 8.0;

    let mut sq_sum: f64 = 0.0;
    for v in a.iter() {
        let d = *v - mean;
        sq_sum += d * d;
    }
    let var = sq_sum / 8.0;
    let std = var.sqrt();

    let lo = mean - 2.0 * std;
    let hi = mean + 2.0 * std;

    for i in 0..8 {
        let inside = (a[i] > lo) && (a[i] < hi);
        let outside: u32 = if inside { 0 } else { 1 };
        assert_eq!(outside, claimed[i]);
    }
}
