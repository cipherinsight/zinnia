// DS-1000 case 441 — Risc0 guest.
// Kernel: `number in a` on Integer[4]; returns bool.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [i64; 4] = [0; 4];
    for k in 0..4 {
        a[k] = env::read();
    }
    let number: i64 = env::read();
    let claimed: u64 = env::read();

    let mut contained: u64 = 0;
    for k in 0..4 {
        let eq: u64 = if a[k] == number { 1 } else { 0 };
        contained = contained | eq;
    }

    assert_eq!(contained, claimed);
}
