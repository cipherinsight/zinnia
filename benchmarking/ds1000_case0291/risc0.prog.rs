// DS-1000 case 291 — Risc0 guest program.
// Kernel: `a.shape` on Integer[2, 5, 5, 2].

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(100);
    for _ in 0..100 {
        a.push(env::read());
    }
    let claim_d0: u64 = env::read();
    let claim_d1: u64 = env::read();
    let claim_d2: u64 = env::read();
    let claim_d3: u64 = env::read();

    let d0: u64 = 2;
    let d1: u64 = 5;
    let d2: u64 = 5;
    let d3: u64 = 2;

    assert_eq!(d0, claim_d0);
    assert_eq!(d1, claim_d1);
    assert_eq!(d2, claim_d2);
    assert_eq!(d3, claim_d3);
    let _ = a.len();
}
