// DS-1000 case 491 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 6;
    const ATOL: f64 = 1.0e-8;
    const RTOL: f64 = 1.0e-5;
    let mut s1: [f64; N] = [0.0; N];
    for k in 0..N { s1[k] = env::read(); }
    let mut s2: [f64; N] = [0.0; N];
    for k in 0..N { s2[k] = env::read(); }
    let claimed: i64 = env::read();

    let mut count: i64 = 0;
    for k in 0..N {
        let diff = (s1[k] - s2[k]).abs();
        let bound = ATOL + RTOL * s2[k].abs();
        let close: bool = diff <= bound;
        let not_close: i64 = if close { 0 } else { 1 };
        count += not_close;
    }

    assert_eq!(count, claimed);
}
