// DS-1000 case 433 — SP1 guest.
// Kernel: a[zero_rows, :] = 0; a[:, zero_cols] = 0 with scalar zero_rows,
// zero_cols on Integer[4, 4]. Each cell becomes 0 iff its row equals
// zero_rows or its column equals zero_cols.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { a.push(sp1_zkvm::io::read::<i64>()); }
    let zero_rows: i64 = sp1_zkvm::io::read::<i64>();
    let zero_cols: i64 = sp1_zkvm::io::read::<i64>();
    let mut claimed: Vec<i64> = Vec::with_capacity(16);
    for _ in 0..16 { claimed.push(sp1_zkvm::io::read::<i64>()); }

    for i in 0..4 {
        for j in 0..4 {
            let row_match = (i as i64) == zero_rows;
            let col_match = (j as i64) == zero_cols;
            let out: i64 = if row_match || col_match { 0 } else { a[i * 4 + j] };
            assert_eq!(out, claimed[i * 4 + j]);
        }
    }
}
