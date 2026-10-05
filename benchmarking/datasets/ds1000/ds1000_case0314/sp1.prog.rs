// DS-1000 case 314 — SP1 guest program.
// Kernel:
//   idx = unravel(a.argmax(), shape)
//   a[idx] = a.min()
//   result = unravel(a.argmax(), shape)
// All in C-order on Float[8, 6].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        a.push(row);
    }
    let claim_row: u64 = sp1_zkvm::io::read::<u64>();
    let claim_col: u64 = sp1_zkvm::io::read::<u64>();

    // First argmax (C-order flat).
    let mut best_val: f64 = f64::NEG_INFINITY;
    let mut best_idx: u64 = 0;
    for i in 0..8 {
        for j in 0..6 {
            if a[i][j] > best_val {
                best_val = a[i][j];
                best_idx = (i * 6 + j) as u64;
            }
        }
    }
    let idx_row: usize = (best_idx / 6) as usize;
    let idx_col: usize = (best_idx % 6) as usize;

    // a.min().
    let mut min_val: f64 = f64::INFINITY;
    for i in 0..8 {
        for j in 0..6 {
            if a[i][j] < min_val {
                min_val = a[i][j];
            }
        }
    }
    a[idx_row][idx_col] = min_val;

    // Second argmax → unravel.
    let mut best_val2: f64 = f64::NEG_INFINITY;
    let mut best_idx2: u64 = 0;
    for i in 0..8 {
        for j in 0..6 {
            if a[i][j] > best_val2 {
                best_val2 = a[i][j];
                best_idx2 = (i * 6 + j) as u64;
            }
        }
    }
    let row: u64 = best_idx2 / 6;
    let col: u64 = best_idx2 % 6;

    assert_eq!(row, claim_row);
    assert_eq!(col, claim_col);
}
