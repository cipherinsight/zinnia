// DS-1000 case 0411 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    const R: i64 = 5;
    let low: i64 = env::read();
    let high: i64 = env::read();
    let mut a = [[0.0f64; 5]; 5];
    for i in 0..5usize { for j in 0..5usize { a[i][j] = env::read(); } }
    let h = (high - low) as usize;
    let mut out = vec![vec![0.0f64; h]; h];
    for i in 0..h {
        for j in 0..h {
            let si = low + i as i64;
            let sj = low + j as i64;
            if si >= 0 && si < R && sj >= 0 && sj < R {
                out[i][j] = a[si as usize][sj as usize];
            }
        }
    }
    for row in &out {
        for v in row {
            let claim: f64 = env::read();
            assert!((v - claim).abs() < 1e-9);
        }
    }
}
