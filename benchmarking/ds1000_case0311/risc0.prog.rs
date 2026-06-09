// DS-1000 case 311 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        data.push(row);
    }
    let claim_row: u64 = env::read();
    let claim_col: u64 = env::read();

    let mut best_val: f64 = f64::NEG_INFINITY;
    let mut best_idx: u64 = 0;
    for i in 0..8 {
        for j in 0..6 {
            if data[i][j] > best_val {
                best_val = data[i][j];
                best_idx = (i * 6 + j) as u64;
            }
        }
    }

    let row: u64 = best_idx % 8;
    let col: u64 = best_idx / 8;

    assert_eq!(row, claim_row);
    assert_eq!(col, claim_col);
}
