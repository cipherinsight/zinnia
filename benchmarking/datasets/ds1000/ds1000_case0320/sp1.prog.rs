// DS-1000 case 320 — SP1 guest program.
// `np.unravel_index(a.argmin(), a.shape)` on Float[5, 6] → (row, col).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(5);
    for _ in 0..5 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(sp1_zkvm::io::read::<f64>());
        }
        data.push(row);
    }
    let claimed_row: u64 = sp1_zkvm::io::read::<u64>();
    let claimed_col: u64 = sp1_zkvm::io::read::<u64>();

    let mut best_val: f64 = data[0][0];
    let mut best_idx: u64 = 0;
    for k in 1..30u64 {
        let i = (k / 6) as usize;
        let j = (k % 6) as usize;
        if data[i][j] < best_val {
            best_val = data[i][j];
            best_idx = k;
        }
    }
    let row = best_idx / 6;
    let col = best_idx % 6;

    assert_eq!(row, claimed_row);
    assert_eq!(col, claimed_col);
}
