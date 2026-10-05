// DS-1000 case 0407 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 100;
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(env::read()); }
    let mut accmap: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { accmap.push(env::read()); }
    let out_len: u32 = env::read();
    let add: i64 = *accmap.iter().max().unwrap();
    for v in &mut accmap { if *v < 0 { *v += add + 1; } }
    let mut result = vec![0i64; out_len as usize];
    for i in 0..N { result[accmap[i] as usize] += a[i]; }
    for k in 0..out_len as usize {
        let claim: i64 = env::read();
        assert_eq!(result[k], claim);
    }
}
