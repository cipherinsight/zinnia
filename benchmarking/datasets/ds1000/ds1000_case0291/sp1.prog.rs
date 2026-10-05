// DS-1000 case 291 — SP1 guest program.
// Kernel: `a.shape` on Integer[2, 5, 5, 2].
// Reads the 100 integer cells (witnessed but unused) and asserts the 4-tuple shape.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(100);
    for _ in 0..100 {
        a.push(sp1_zkvm::io::read::<i64>());
    }
    let claim_d0: u64 = sp1_zkvm::io::read::<u64>();
    let claim_d1: u64 = sp1_zkvm::io::read::<u64>();
    let claim_d2: u64 = sp1_zkvm::io::read::<u64>();
    let claim_d3: u64 = sp1_zkvm::io::read::<u64>();

    let d0: u64 = 2;
    let d1: u64 = 5;
    let d2: u64 = 5;
    let d3: u64 = 2;

    assert_eq!(d0, claim_d0);
    assert_eq!(d1, claim_d1);
    assert_eq!(d2, claim_d2);
    assert_eq!(d3, claim_d3);
    // Witness a so it isn't optimised out.
    let _ = a.len();
}
