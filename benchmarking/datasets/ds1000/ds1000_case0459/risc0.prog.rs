// DS-1000 case 459 — Risc0 guest. mean of NA cast to float.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [f64; 5] = [0.0; 5];
    for k in 0..5 { a[k] = env::read(); }
    let _a = a;

    let mut na: [i64; 5] = [0; 5];
    for k in 0..5 { na[k] = env::read(); }
    let claimed: f64 = env::read();

    let mut na_f: [f64; 5] = [0.0; 5];
    for k in 0..5 { na_f[k] = na[k] as f64; }

    let mut sum: f64 = na_f[0];
    for k in 1..5 { sum = sum + na_f[k]; }
    let n: f64 = 5.0;
    let mean: f64 = sum / n;

    assert!((mean - claimed).abs() < 0.001);
}
