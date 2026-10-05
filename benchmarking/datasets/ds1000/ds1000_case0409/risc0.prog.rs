// DS-1000 case 409 — Risc0 guest program. z = x + y for Float[8, 6].

use risc0_zkvm::guest::env;

fn main() {
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(env::read()); }
        x.push(row);
    }
    let mut y: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(env::read()); }
        y.push(row);
    }
    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row = Vec::with_capacity(6);
        for _ in 0..6 { row.push(env::read()); }
        claimed.push(row);
    }

    for i in 0..8 {
        for j in 0..6 {
            let z: f64 = x[i][j] + y[i][j];
            assert!((z - claimed[i][j]).abs() < 1e-6);
        }
    }
}
