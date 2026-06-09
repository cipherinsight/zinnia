// DS-1000 case 427 — SP1 guest program.
// Kernel: res = 0; for i in a: res ^= i. Then output = bit decomp of res (m MSB-first bits).
// Inputs: a: Integer[20], m: Integer.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let m: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<u64> = Vec::with_capacity(20);
    for _ in 0..20 { a.push(sp1_zkvm::io::read::<u64>()); }
    let mut res: u64 = 0;
    for v in &a { res ^= v; }
    for j in 0..m as usize {
        let shift = (m as u32) - 1 - (j as u32);
        let bit: u64 = if (res & (1u64 << shift)) > 0 { 1 } else { 0 };
        let claim: u64 = sp1_zkvm::io::read::<u64>();
        assert_eq!(bit, claim);
    }
}
