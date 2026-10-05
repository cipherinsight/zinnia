// DS-1000 case 322 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut data: Vec<Vec<i64>> = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut row: Vec<i64> = Vec::with_capacity(2);
        for _ in 0..2 {
            row.push(env::read());
        }
        data.push(row);
    }
    let claimed_row: u64 = env::read();
    let claimed_col: u64 = env::read();

    let mut min_val: i64 = data[0][0];
    for i in 0..2usize {
        for j in 0..2usize {
            if data[i][j] < min_val {
                min_val = data[i][j];
            }
        }
    }
    let mut out_row: u64 = 0;
    let mut out_col: u64 = 0;
    let mut found: bool = false;
    for i in 0..2usize {
        for j in 0..2usize {
            if !found && data[i][j] == min_val {
                out_row = i as u64;
                out_col = j as u64;
                found = true;
            }
        }
    }
    assert!(found);

    assert_eq!(out_row, claimed_row);
    assert_eq!(out_col, claimed_col);
}
