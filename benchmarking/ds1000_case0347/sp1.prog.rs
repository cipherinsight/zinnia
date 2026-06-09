// DS-1000 case 347 — SP1 guest program.
// Kernel: a[row-1, :] /= divide_number; result = np.multiply.reduce(a[row-1, :])
// On Float[8, 6] with runtime row, divide_number. Output: scalar float.

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
    let row: i64 = sp1_zkvm::io::read::<i64>();
    let divide_number: i64 = sp1_zkvm::io::read::<i64>();
    let claimed: f64 = sp1_zkvm::io::read::<f64>();

    let row_idx: usize = (row - 1) as usize;
    let dn = divide_number as f64;

    let mut row_vals: [f64; 6] = [0.0; 6];
    for j in 0..6 {
        row_vals[j] = a[row_idx][j] / dn;
    }

    let mut result: f64 = row_vals[0];
    for k in 1..6 {
        result = result * row_vals[k];
    }

    assert!((result - claimed).abs() < 1e-3);
}
