// DS-1000 case 0299 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 200;
    let cols: u32 = env::read();
    let mut flat: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { flat.push(env::read()); }
    let m = *flat.iter().min().unwrap();
    let temp: Vec<i64> = flat.iter().map(|v| v - m).collect();
    let mut b = vec![vec![0u64; cols as usize]; N];
    for i in 0..N { b[i][temp[i] as usize] = 1; }
    for row in &b {
        for v in row {
            let claim: u64 = env::read();
            assert_eq!(*v, claim);
        }
    }
}
