// DS-1000 case 309 — SP1 guest program.
// Reads an 8×6 f64 matrix and a claimed argmax index from stdin,
// recomputes the argmax, and asserts equality.

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
    let result: u64 = sp1_zkvm::io::read::<u64>();

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

    assert_eq!(best_idx, result);
}
