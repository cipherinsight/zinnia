// DS-1000 case 0296 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 50;
    let cols: u32 = env::read();
    let mut a: Vec<u64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(env::read()); }
    let mut b = vec![vec![0u64; cols as usize]; N];
    for i in 0..N { b[i][a[i] as usize] = 1; }
    for row in &b {
        for v in row {
            let claim: u64 = env::read();
            assert_eq!(*v, claim);
        }
    }
}
