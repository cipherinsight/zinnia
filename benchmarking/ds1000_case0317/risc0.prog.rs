// DS-1000 case 317 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<f64> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        a.push(env::read());
    }

    let mut out: Vec<f64> = Vec::with_capacity(1000);
    for i in 0..1000 {
        out.push(a[i]);
    }

    for i in 0..1000 {
        let claim_v: f64 = env::read();
        assert!((out[i] - claim_v).abs() < 1e-12);
    }
}
