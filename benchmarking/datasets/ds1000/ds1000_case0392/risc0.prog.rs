// DS-1000 case 392 — Risc0 guest. a[low:high, :].

use risc0_zkvm::guest::env;

fn main() {
    const ROWS: usize = 20;
    const COLS: usize = 10;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let mut row: Vec<f64> = Vec::with_capacity(COLS);
        for _ in 0..COLS { row.push(env::read()); }
        a.push(row);
    }
    let low: u32 = env::read();
    let high: u32 = env::read();
    let lo = low as usize;
    let hi = high as usize;
    let height = hi - lo;

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(height);
    for _ in 0..height {
        let mut row: Vec<f64> = Vec::with_capacity(COLS);
        for _ in 0..COLS { row.push(env::read()); }
        claimed.push(row);
    }

    for i in 0..height {
        for j in 0..COLS {
            let v = a[lo + i][j];
            assert!((v - claimed[i][j]).abs() < 0.001);
        }
    }
}
