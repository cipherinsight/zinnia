// DS-1000 case 345 — Risc0 guest program.

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
    let col: i64 = env::read();
    let multiply_number: i64 = env::read();
    let mut claimed: [f64; 8] = [0.0; 8];
    for i in 0..8 {
        claimed[i] = env::read();
    }

    let col_idx: usize = (col - 1) as usize;
    let mn = multiply_number as f64;
    let mut col_vals: [f64; 8] = [0.0; 8];
    for i in 0..8 {
        col_vals[i] = a[i][col_idx] * mn;
    }

    let mut result: [f64; 8] = [0.0; 8];
    result[0] = col_vals[0];
    for k in 1..8 {
        result[k] = result[k - 1] + col_vals[k];
    }

    for i in 0..8 {
        assert!((result[i] - claimed[i]).abs() < 1e-3);
    }
}
