// DS-1000 case 332 — gcd of two scalar integers.
// kernel.py: gcd(numerator, denominator); return (numerator/gcd, denominator/gcd).
// Inputs: numerator=24, denominator=18; expected (4, 3).

fn gcd(a: u128, b: u128) -> u128 {
    let mut x: u128 = a;
    let mut y: u128 = b;
    while y != 0 {
        let t = y;
        y = x % y;
        x = t;
    };
    x
}

#[executable]
pub fn main() {
    let num: u128 = 24;
    let den: u128 = 18;
    let claim_n: u128 = 4;
    let claim_d: u128 = 3;
    let g = gcd(num, den);
    assert!(num / g == claim_n);
    assert!(den / g == claim_d);
}
