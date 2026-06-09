// DS-1000 case 480 — SP1 guest program.
// kernel.py:
//     result = ((x == a) & (y == b)).argmax()
//     if x[result] != a or y[result] != b: result = -1
// On Integer[8] × Integer[8] + two integer scalars.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut x: [i64; 8] = [0; 8];
    for k in 0..8 { x[k] = sp1_zkvm::io::read::<i64>(); }
    let mut y: [i64; 8] = [0; 8];
    for k in 0..8 { y[k] = sp1_zkvm::io::read::<i64>(); }
    let a: i64 = sp1_zkvm::io::read::<i64>();
    let b: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: i64 = sp1_zkvm::io::read::<i64>();

    // Build per-index mask.
    let mut mask: [u64; 8] = [0; 8];
    for k in 0..8 {
        let xa: bool = x[k] == a;
        let yb: bool = y[k] == b;
        mask[k] = if xa && yb { 1 } else { 0 };
    }

    // argmax: index of first maximum (argmax of bool ties on the first 1, or 0
    // if all zero).
    let mut best_val: u64 = mask[0];
    let mut best_idx: i64 = 0;
    for k in 1..8 {
        if mask[k] > best_val {
            best_val = mask[k];
            best_idx = k as i64;
        }
    }

    // If not found, return -1.
    let mut result: i64 = best_idx;
    if x[best_idx as usize] != a || y[best_idx as usize] != b {
        result = -1;
    }

    assert_eq!(result, claimed);
}
