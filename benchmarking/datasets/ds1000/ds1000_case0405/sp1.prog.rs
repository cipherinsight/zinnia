// DS-1000 case 405 — SP1 guest program.
// Kernel: np.bincount(accmap, weights=a). Inputs: a, accmap each Integer[100].
// Output length is max(accmap)+1 (data-dependent). We accept this length from the host.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: u32 = sp1_zkvm::io::read::<u32>();
    let out_len: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<i64> = Vec::with_capacity(n as usize);
    for _ in 0..n { a.push(sp1_zkvm::io::read::<i64>()); }
    let mut accmap: Vec<i64> = Vec::with_capacity(n as usize);
    for _ in 0..n { accmap.push(sp1_zkvm::io::read::<i64>()); }
    let mut result = vec![0i64; out_len as usize];
    for i in 0..n as usize {
        let k = accmap[i] as usize;
        result[k] += a[i];
    }
    for k in 0..out_len as usize {
        let claim: i64 = sp1_zkvm::io::read::<i64>();
        assert_eq!(result[k], claim);
    }
}
