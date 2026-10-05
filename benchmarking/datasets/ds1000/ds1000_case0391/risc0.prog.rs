// DS-1000 case 391 — Risc0 guest. a[:, low:high].

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
    let width = hi - lo;

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(ROWS);
    for _ in 0..ROWS {
        let mut row: Vec<f64> = Vec::with_capacity(width);
        for _ in 0..width { row.push(env::read()); }
        claimed.push(row);
    }

    for i in 0..ROWS {
        for k in 0..width {
            let v = a[i][lo + k];
            assert!((v - claimed[i][k]).abs() < 0.001);
        }
    }
}
