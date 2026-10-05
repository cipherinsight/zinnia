// DS-1000 case 0471 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    let mut z = vec![vec![vec![0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { z[i][j][k] = env::read(); } } }
    let mut out: Vec<f64> = Vec::with_capacity(N * M);
    for i in 0..N { for j in 0..M { out.push(z[i][j][K - 1]); } }
    for v in &out {
        let claim: f64 = env::read();
        assert!((v - claim).abs() < 1e-9);
    }
}
