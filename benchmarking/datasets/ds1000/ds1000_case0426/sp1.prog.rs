// DS-1000 case 426 — SP1 guest program.
// Kernel: ((a[:,None] & (1 << np.arange(m))[::-1]) > 0).astype(int)
// Inputs: a: Integer[20], m: Integer.
// Output: 20×m matrix of 0/1.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let m: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<u64> = Vec::with_capacity(20);
    for _ in 0..20 { a.push(sp1_zkvm::io::read::<u64>()); }
    // Recompute.
    for i in 0..20usize {
        for j in 0..m as usize {
            let shift = (m as u32) - 1 - (j as u32);
            let mask: u64 = 1u64 << shift;
            let bit: u64 = if (a[i] & mask) > 0 { 1 } else { 0 };
            let claim: u64 = sp1_zkvm::io::read::<u64>();
            assert_eq!(bit, claim);
        }
    }
}
