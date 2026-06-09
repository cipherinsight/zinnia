// DS-1000 case 331 — SP1 guest program.
// Kernel: gcd(numerator, denominator); return (numerator/gcd, denominator/gcd).

#![no_main]
sp1_zkvm::entrypoint!(main);

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

pub fn main() {
    let num: u64 = sp1_zkvm::io::read::<u64>();
    let den: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_n: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_d: u64 = sp1_zkvm::io::read::<u64>();
    let g = gcd(num, den);
    assert_eq!(num / g, claimed_n);
    assert_eq!(den / g, claimed_d);
}
