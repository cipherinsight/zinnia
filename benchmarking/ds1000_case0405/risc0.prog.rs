// DS-1000 case 405 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let n: u32 = env::read();
    let out_len: u32 = env::read();
    let mut a: Vec<i64> = Vec::with_capacity(n as usize);
    for _ in 0..n { a.push(env::read()); }
    let mut accmap: Vec<i64> = Vec::with_capacity(n as usize);
    for _ in 0..n { accmap.push(env::read()); }
    let mut result = vec![0i64; out_len as usize];
    for i in 0..n as usize { result[accmap[i] as usize] += a[i]; }
    for k in 0..out_len as usize {
        let claim: i64 = env::read();
        assert_eq!(result[k], claim);
    }
}
