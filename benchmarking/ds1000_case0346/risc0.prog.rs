// DS-1000 case 346 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 {
            row.push(env::read());
        }
        a.push(row);
    }
    let row: i64 = env::read();
    let multiply_number: i64 = env::read();
    let mut claimed: [f64; 6] = [0.0; 6];
    for j in 0..6 {
        claimed[j] = env::read();
    }

    let row_idx: usize = (row - 1) as usize;
    let mn = multiply_number as f64;

    let mut row_vals: [f64; 6] = [0.0; 6];
    for j in 0..6 {
        row_vals[j] = a[row_idx][j] * mn;
    }

    let mut result: [f64; 6] = [0.0; 6];
    result[0] = row_vals[0];
    for k in 1..6 {
        result[k] = result[k - 1] + row_vals[k];
    }

    for j in 0..6 {
        assert!((result[j] - claimed[j]).abs() < 1e-3);
    }
}
