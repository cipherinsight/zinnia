// DS-1000 case 427 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let m: u32 = env::read();
    let mut a: Vec<u64> = Vec::with_capacity(20);
    for _ in 0..20 { a.push(env::read()); }
    let mut res: u64 = 0;
    for v in &a { res ^= v; }
    for j in 0..m as usize {
        let shift = (m as u32) - 1 - (j as u32);
        let bit: u64 = if (res & (1u64 << shift)) > 0 { 1 } else { 0 };
        let claim: u64 = env::read();
        assert_eq!(bit, claim);
    }
}
