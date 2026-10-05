// DS-1000 case 451 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let claimed: f64 = env::read();
    let zero: f64 = 0.0;
    assert!((zero - claimed).abs() < 0.001);
}
