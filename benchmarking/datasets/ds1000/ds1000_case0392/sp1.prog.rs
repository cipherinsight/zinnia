// DS-1000 case 392 — SP1 guest program.
// Kernel: result = a[low:high, :] on Float[20, 10].
// Sample: low=5, high=15 → result shape (10, 10).

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
    let height = hi - lo;

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(height);
    for _ in 0..height {
        let mut row: Vec<f64> = Vec::with_capacity(COLS);
        for _ in 0..COLS { row.push(sp1_zkvm::io::read::<f64>()); }
        claimed.push(row);
    }

    for i in 0..height {
        for j in 0..COLS {
            let v = a[lo + i][j];
            assert!((v - claimed[i][j]).abs() < 0.001);
        }
    }
}
