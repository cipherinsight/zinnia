// DS-1000 case 312 — SP1 guest program.
// Kernel: np.unravel_index(a.argmax(), a.shape) on Float[8, 6].
// C-order unravel: row = idx / 6, col = idx % 6.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        data.push(row);
    }
    let claim_row: u64 = sp1_zkvm::io::read::<u64>();
    let claim_col: u64 = sp1_zkvm::io::read::<u64>();

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

    let row: u64 = best_idx / 6;
    let col: u64 = best_idx % 6;

    assert_eq!(row, claim_row);
    assert_eq!(col, claim_col);
}
