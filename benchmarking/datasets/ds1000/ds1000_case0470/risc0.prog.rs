// DS-1000 case 470 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let a0: f64 = env::read();
    let a1: f64 = env::read();
    let a2: f64 = env::read();
    let claim_re: f64 = env::read();
    let claim_im: f64 = env::read();

    let n: f64 = 3.0;
    let s_re: f64 = a0 + a1 + a2;
    let s_im: f64 = 0.0;
    let res_re: f64 = s_re / n;
    let res_im: f64 = s_im / n;

    assert!((res_re - claim_re).abs() <= 0.001);
    assert!((res_im - claim_im).abs() <= 0.001);
}
