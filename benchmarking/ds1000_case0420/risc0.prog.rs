// DS-1000 case 420 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let x: i64 = env::read();
    let x_min: i64 = env::read();
    let x_max: i64 = env::read();
    let claim: i64 = env::read();
    let inner = 3 * x * x - 2 * x * x * x;
    let mid = if x > x_max { x_max } else { inner };
    let out = if x < x_min { x_min } else { mid };
    assert_eq!(out, claim);
}
