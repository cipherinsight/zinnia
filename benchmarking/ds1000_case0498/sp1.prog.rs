// DS-1000 case 0498 — SP1 guest program.
// Kernel: np.pad(arr, ((0, shape[0]-N), (0, shape[1]-M)), 'constant') on Float[N≤64, M≤64, K≤64].
// We interpret as pad first 2 axes, leave third (K) unchanged. Concrete N=2, M=3, K=2.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 2;
    let new_n: u32 = sp1_zkvm::io::read::<u32>();
    let new_m: u32 = sp1_zkvm::io::read::<u32>();
    let mut arr = vec![vec![vec![0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { arr[i][j][k] = sp1_zkvm::io::read::<f64>(); } } }
    for i in 0..new_n as usize {
        for j in 0..new_m as usize {
            for k in 0..K {
                let v: f64 = if i < N && j < M { arr[i][j][k] } else { 0.0 };
                let claim: f64 = sp1_zkvm::io::read::<f64>();
                assert!((v - claim).abs() < 1e-9);
            }
        }
    }
}
