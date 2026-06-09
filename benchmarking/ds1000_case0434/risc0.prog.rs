// DS-1000 case 434 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { a.push(env::read()); }
    let mut zero_rows: Vec<i64> = Vec::with_capacity(2);
    for _ in 0..2 { zero_rows.push(env::read()); }
    let mut zero_cols: Vec<i64> = Vec::with_capacity(2);
    for _ in 0..2 { zero_cols.push(env::read()); }
    let mut claimed: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { claimed.push(env::read()); }

    for i in 0..4 {
        let i_i = i as i64;
        let row_zero = (i_i == zero_rows[0]) || (i_i == zero_rows[1]);
        for j in 0..4 {
            let j_i = j as i64;
            let col_zero = (j_i == zero_cols[0]) || (j_i == zero_cols[1]);
            let any = row_zero || col_zero;
            let out: i64 = if any { 0 } else { a[i * 4 + j] };
            assert_eq!(out, claimed[i * 4 + j]);
        }
    }
}
