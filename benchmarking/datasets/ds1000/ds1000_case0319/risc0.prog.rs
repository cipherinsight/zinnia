// DS-1000 case 319 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(6);
        for _ in 0..6 {
            let mut row: Vec<f64> = Vec::with_capacity(8);
            for _ in 0..8 {
                row.push(env::read());
            }
            plane.push(row);
        }
        a.push(plane);
    }
    let mut permutation: [i64; 8] = [0; 8];
    for k in 0..8 {
        permutation[k] = env::read();
    }

    let mut c: [i64; 8] = [0; 8];
    for k in 0..8 {
        let p: usize = permutation[k] as usize;
        c[p] = k as i64;
    }

    for i in 0..8 {
        let src_i: usize = c[i] as usize;
        for j in 0..6 {
            for k in 0..8 {
                let claim_v: f64 = env::read();
                assert!((a[src_i][j][k] - claim_v).abs() < 1e-12);
            }
        }
    }
}
