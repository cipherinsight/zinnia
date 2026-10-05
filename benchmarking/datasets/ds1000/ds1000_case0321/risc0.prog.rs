// DS-1000 case 321 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        data.push(row);
    }
    let claimed_row: u64 = env::read();
    let claimed_col: u64 = env::read();

    let mut best_val: f64 = data[0][0];
    let mut best_idx: u64 = 0;
    for k in 1..30u64 {
        let i = (k / 6) as usize;
        let j = (k % 6) as usize;
        if data[i][j] > best_val {
            best_val = data[i][j];
            best_idx = k;
        }
    }
    let row = best_idx / 6;
    let col = best_idx % 6;

    assert_eq!(row, claimed_row);
    assert_eq!(col, claimed_col);
}
