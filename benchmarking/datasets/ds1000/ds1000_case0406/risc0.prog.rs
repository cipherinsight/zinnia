// DS-1000 case 0406 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 100;
    let out_len: u32 = env::read();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(env::read()); }
    let mut index: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { index.push(env::read()); }
    let mut result = vec![0i64; out_len as usize];
    let mut seen = vec![false; out_len as usize];
    for i in 0..N {
        let k = index[i] as usize;
        if !seen[k] || a[i] > result[k] {
            result[k] = a[i];
            seen[k] = true;
        }
    }
    for k in 0..out_len as usize {
        let claim: i64 = env::read();
        let v: i64 = if seen[k] { result[k] } else { 0 };
        assert_eq!(v, claim);
    }
}
