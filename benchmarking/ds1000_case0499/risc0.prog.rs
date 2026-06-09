// DS-1000 case 0499 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const R: usize = 42;
    const C: usize = 10;
    let new_rows: u32 = env::read();
    let new_cols: u32 = env::read();
    let mut a: Vec<f64> = Vec::with_capacity(R * C);
    for _ in 0..(R * C) { a.push(env::read()); }
    for i in 0..new_rows as usize {
        for j in 0..new_cols as usize {
            let v: f64 = if i < R && j < C { a[i * C + j] } else { 0.0 };
            let claim: f64 = env::read();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
