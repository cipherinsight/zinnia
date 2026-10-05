// DS-1000 case 0406 — SP1 guest program.
// Kernel: for each unique k in index: result[k] = max(a where index==k). Other bins stay 0.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 100;
    let out_len: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<i64>()); }
    let mut index: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { index.push(sp1_zkvm::io::read::<i64>()); }
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
