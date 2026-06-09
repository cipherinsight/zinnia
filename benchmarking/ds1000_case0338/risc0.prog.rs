// DS-1000 case 338 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(8);
        for _ in 0..8 {
            row.push(env::read());
        }
        a.push(row);
    }
    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut row: Vec<f64> = Vec::with_capacity(8);
        for _ in 0..8 {
            row.push(env::read());
        }
        claimed.push(row);
    }

    for i in 0..8usize {
        let d = a[i][i];
        assert!((d - claimed[0][i]).abs() < 1e-9);
    }
    for i in 0..8usize {
        let f = a[i][7 - i];
        assert!((f - claimed[1][i]).abs() < 1e-9);
    }
}
