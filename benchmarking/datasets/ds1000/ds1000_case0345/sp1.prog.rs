// DS-1000 case 345 — SP1 guest program.
// Kernel: a[:, col-1] *= multiply_number; result = cumsum(a[:, col-1])
// On Float[8, 6] with runtime col, multiply_number. Output: 8 floats.

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
    let col: i64 = sp1_zkvm::io::read::<i64>();
    let multiply_number: i64 = sp1_zkvm::io::read::<i64>();

    let mut claimed: [f64; 8] = [0.0; 8];
    for i in 0..8 {
        claimed[i] = sp1_zkvm::io::read::<f64>();
    }

    let col_idx: usize = (col - 1) as usize;

    // a[:, col-1] *= multiply_number
    let mn = multiply_number as f64;
    let mut col_vals: [f64; 8] = [0.0; 8];
    for i in 0..8 {
        col_vals[i] = a[i][col_idx] * mn;
    }

    // cumsum
    let mut result: [f64; 8] = [0.0; 8];
    result[0] = col_vals[0];
    for k in 1..8 {
        result[k] = result[k - 1] + col_vals[k];
    }

    for i in 0..8 {
        assert!((result[i] - claimed[i]).abs() < 1e-3);
    }
}
