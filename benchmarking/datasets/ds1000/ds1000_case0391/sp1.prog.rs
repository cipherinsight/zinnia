// DS-1000 case 391 — SP1 guest program.
// Kernel: result = a[:, low:high] on Float[20, 10].
// Sample: low=3, high=7 → result shape (20, 4).

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const ROWS: usize = 20;
    const COLS: usize = 10;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let mut row: Vec<f64> = Vec::with_capacity(COLS);
        for _ in 0..COLS { row.push(sp1_zkvm::io::read::<f64>()); }
        a.push(row);
    }
    let low: u32 = sp1_zkvm::io::read::<u32>();
    let high: u32 = sp1_zkvm::io::read::<u32>();
    let lo = low as usize;
    let hi = high as usize;
    let width = hi - lo;

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let mut row: Vec<f64> = Vec::with_capacity(width);
        for _ in 0..width { row.push(sp1_zkvm::io::read::<f64>()); }
        claimed.push(row);
    }

    for i in 0..ROWS {
        for k in 0..width {
            let v = a[i][lo + k];
            assert!((v - claimed[i][k]).abs() < 0.001);
        }
    }
}
