// DS-1000 case 413 — Risc0 guest program.
// `x[x.imag != 0]` on a real Float[10]; mask is empty.

use risc0_zkvm::guest::env;

fn main() {
    let mut x: [f64; 10] = [0.0; 10];
    for k in 0..10 {
        x[k] = env::read();
    }
    let _result: Vec<f64> = Vec::new();
    let mut s: f64 = 0.0;
    for k in 0..10 { s += x[k] * 0.0; }
    assert!(s == 0.0);
}
