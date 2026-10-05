// DS-1000 case 503 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<Vec<f64>>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut plane: Vec<Vec<f64>> = Vec::with_capacity(13);
        for _ in 0..13 {
            let mut row: Vec<f64> = Vec::with_capacity(5);
            for _ in 0..5 {
                row.push(env::read());
            }
            plane.push(row);
        }
        a.push(plane);
    }
    let mut b: Vec<Vec<u64>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut row: Vec<u64> = Vec::with_capacity(13);
        for _ in 0..13 {
            row.push(env::read());
        }
        b.push(row);
    }
    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(13);
    for _ in 0..13 {
        let mut row: Vec<f64> = Vec::with_capacity(13);
        for _ in 0..13 {
            row.push(env::read());
        }
        claimed.push(row);
    }

    for i in 0..13 {
        for j in 0..13 {
            let k = b[i][j] as usize;
            let v = a[i][j][k];
            assert!((v - claimed[i][j]).abs() < 1e-6);
        }
    }
}
