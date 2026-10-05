// DS-1000 case 0498 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 2;
    let new_n: u32 = env::read();
    let new_m: u32 = env::read();
    let mut arr = vec![vec![vec![0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { arr[i][j][k] = env::read(); } } }
    for i in 0..new_n as usize {
        for j in 0..new_m as usize {
            for k in 0..K {
                let v: f64 = if i < N && j < M { arr[i][j][k] } else { 0.0 };
                let claim: f64 = env::read();
                assert!((v - claim).abs() < 1e-9);
            }
        }
    }
}
