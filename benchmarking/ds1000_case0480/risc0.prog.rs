// DS-1000 case 480 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut x: [i64; 8] = [0; 8];
    for k in 0..8 { x[k] = env::read(); }
    let mut y: [i64; 8] = [0; 8];
    for k in 0..8 { y[k] = env::read(); }
    let a: i64 = env::read();
    let b: i64 = env::read();
    let claimed: i64 = env::read();

    let mut mask: [u64; 8] = [0; 8];
    for k in 0..8 {
        let xa: bool = x[k] == a;
        let yb: bool = y[k] == b;
        mask[k] = if xa && yb { 1 } else { 0 };
    }

    let mut best_val: u64 = mask[0];
    let mut best_idx: i64 = 0;
    for k in 1..8 {
        if mask[k] > best_val {
            best_val = mask[k];
            best_idx = k as i64;
        }
    }

    let mut result: i64 = best_idx;
    if x[best_idx as usize] != a || y[best_idx as usize] != b {
        result = -1;
    }

    assert_eq!(result, claimed);
}
