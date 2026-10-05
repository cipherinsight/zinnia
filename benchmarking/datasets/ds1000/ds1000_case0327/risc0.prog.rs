// DS-1000 case 327 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let length: u32 = env::read();
    let n: u32 = 10;
    let mut a: Vec<f64> = Vec::with_capacity(n as usize);
    for _ in 0..n { a.push(env::read()); }
    let mut out: Vec<f64> = Vec::with_capacity(length as usize);
    for i in 0..length as usize {
        if i < n as usize { out.push(a[i]); } else { out.push(0.0); }
    }
    for v in &out {
        let claim: f64 = env::read();
        assert!((v - claim).abs() < 1e-9);
    }
}
