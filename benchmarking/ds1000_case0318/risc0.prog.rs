// DS-1000 case 318 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = env::read();
        }
    }
    let mut permutation: [i64; 6] = [0; 6];
    for k in 0..6 {
        permutation[k] = env::read();
    }

    let mut c: [i64; 6] = [0; 6];
    for k in 0..6 {
        let p: usize = permutation[k] as usize;
        c[p] = k as i64;
    }

    let mut out: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            let src_col: usize = c[j] as usize;
            out[i][j] = a[i][src_col];
        }
    }

    for i in 0..8 {
        for j in 0..6 {
            let claim_v: f64 = env::read();
            assert!((out[i][j] - claim_v).abs() < 1e-12);
        }
    }
}
