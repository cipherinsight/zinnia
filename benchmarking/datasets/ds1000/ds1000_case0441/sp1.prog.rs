// DS-1000 case 441 — SP1 guest.
// Kernel: `number in a` on Integer[4]; returns bool.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: [i64; 4] = [0; 4];
    for k in 0..4 {
        a[k] = sp1_zkvm::io::read::<i64>();
    }
    let number: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: u64 = sp1_zkvm::io::read::<u64>();

    let mut contained: u64 = 0;
    for k in 0..4 {
        let eq: u64 = if a[k] == number { 1 } else { 0 };
        contained = contained | eq;
    }

    assert_eq!(contained, claimed);
}
