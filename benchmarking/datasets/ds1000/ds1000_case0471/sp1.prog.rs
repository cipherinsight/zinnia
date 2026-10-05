// DS-1000 case 0471 — SP1 guest program.
// Kernel: Z[..., -1:] — take last element along the final axis.
// Z: Float[N<=64, M<=64, K<=64]. We use concrete dims N=2, M=3, K=4.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    let mut z = vec![vec![vec![0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { z[i][j][k] = sp1_zkvm::io::read::<f64>(); } } }
    let mut out: Vec<f64> = Vec::with_capacity(N * M);
    for i in 0..N { for j in 0..M { out.push(z[i][j][K - 1]); } }
    for v in &out {
        let claim: f64 = sp1_zkvm::io::read::<f64>();
        assert!((v - claim).abs() < 1e-9);
    }
}
