// DS-1000 case 337 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(10);
        for _ in 0..10 {
            row.push(env::read());
        }
        a.push(row);
    }
    let mut claimed: Vec<f64> = Vec::with_capacity(8);
    for _ in 0..8 {
        claimed.push(env::read());
    }

    for i in 0..8usize {
        let v = a[i][9 - i];
        assert!((v - claimed[i]).abs() < 1e-9);
    }
}
