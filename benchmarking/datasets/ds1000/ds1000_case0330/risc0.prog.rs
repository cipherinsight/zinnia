// DS-1000 case 330 — Risc0 guest. `a ** power` on Float[8, 6].

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        a.push(row);
    }
    let power: u64 = env::read();

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        claimed.push(row);
    }

    for i in 0..8usize {
        for j in 0..6usize {
            let mut acc: f64 = 1.0;
            for _ in 0..power {
                acc *= a[i][j];
            }
            assert!((acc - claimed[i][j]).abs() < 1e-9);
        }
    }
}
