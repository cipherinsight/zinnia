// DS-1000 case 0440 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 3;
    let mut y = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K { y[i][j][k] = env::read(); } } }
    let mut x = [[0.0f64; N]; M];
    for i in 0..N {
        for j in 0..M {
            x[j][i] += y[i][j][j].sqrt();
        }
    }
    for j in 0..M {
        for i in 0..N {
            let claim: f64 = env::read();
            assert!((x[j][i] - claim).abs() < 1e-9);
        }
    }
}
