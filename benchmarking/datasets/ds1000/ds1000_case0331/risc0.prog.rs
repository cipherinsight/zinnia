// DS-1000 case 331 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn main() {
    let num: u64 = env::read();
    let den: u64 = env::read();
    let claim_n: u64 = env::read();
    let claim_d: u64 = env::read();
    let g = gcd(num, den);
    assert_eq!(num / g, claim_n);
    assert_eq!(den / g, claim_d);
}
