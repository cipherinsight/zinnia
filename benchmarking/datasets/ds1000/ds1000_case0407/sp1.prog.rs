// DS-1000 case 0407 — SP1 guest program.
// Kernel: add = max(accmap); accmap[accmap<0] += add+1; result = bincount(accmap, weights=a).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 100;
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<i64>()); }
    let mut accmap: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { accmap.push(sp1_zkvm::io::read::<i64>()); }
    let out_len: u32 = sp1_zkvm::io::read::<u32>();
    let add: i64 = *accmap.iter().max().unwrap();
    for v in &mut accmap { if *v < 0 { *v += add + 1; } }
    let mut result = vec![0i64; out_len as usize];
    for i in 0..N { result[accmap[i] as usize] += a[i]; }
    for k in 0..out_len as usize {
        let claim: i64 = sp1_zkvm::io::read::<i64>();
        assert_eq!(result[k], claim);
    }
}
