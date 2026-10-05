// DS-1000 case 0297 — SP1 guest program.
// Kernel: temp = a - a.min(); b = np.zeros((a.size, temp.max()+1)); b[arange, temp] = 1.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 50;
    let cols: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<i64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<i64>()); }
    let m = *a.iter().min().unwrap();
    let temp: Vec<i64> = a.iter().map(|v| v - m).collect();
    let mut b = vec![vec![0u64; cols as usize]; N];
    for i in 0..N { b[i][temp[i] as usize] = 1; }
    for row in &b {
        for v in row {
            let claim: u64 = sp1_zkvm::io::read::<u64>();
            assert_eq!(*v, claim);
        }
    }
}
