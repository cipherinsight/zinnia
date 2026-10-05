// DS-1000 case 0408 — SP1 guest program.
// Kernel: add = max(index); index[index<0] += add+1; for k in unique(index): result[k] = max(a[index==k]).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 100;
    let out_len: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<i64>()); }
    let mut index: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { index.push(sp1_zkvm::io::read::<i64>()); }
    let add: i64 = *index.iter().max().unwrap();
    for v in &mut index { if *v < 0 { *v += add + 1; } }
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
        let claim: i64 = sp1_zkvm::io::read::<i64>();
        let v: i64 = if seen[k] { result[k] } else { 0 };
        assert_eq!(v, claim);
    }
}
