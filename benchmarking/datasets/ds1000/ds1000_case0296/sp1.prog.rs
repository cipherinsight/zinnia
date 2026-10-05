// DS-1000 case 0296 — SP1 guest program.
// Kernel: b = np.zeros((a.size, a.max()+1)); b[np.arange(a.size), a] = 1.
// a: Integer[50]. Output shape: (50, max(a)+1) — dynamic; host supplies max+1.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 50;
    let cols: u32 = sp1_zkvm::io::read::<u32>();
    let mut a: Vec<u64> = Vec::with_capacity(N);
    for _ in 0..N { a.push(sp1_zkvm::io::read::<u64>()); }
    let mut b = vec![vec![0u64; cols as usize]; N];
    for i in 0..N { b[i][a[i] as usize] = 1; }
    for row in &b {
        for v in row {
            let claim: u64 = sp1_zkvm::io::read::<u64>();
            assert_eq!(*v, claim);
        }
    }
}
