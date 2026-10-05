// DS-1000 case 416 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 { row.push(env::read()); }
        data.push(row);
    }
    let bin_size: u32 = env::read();
    let trimmed: usize = ((6u32 / bin_size) * bin_size) as usize;
    let n_bins: usize = trimmed / (bin_size as usize);

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(n_bins);
        for _ in 0..n_bins { row.push(env::read()); }
        claimed.push(row);
    }

    let bs_f = bin_size as f64;
    for i in 0..8 {
        for b in 0..n_bins {
            let mut s: f64 = 0.0;
            for k in 0..(bin_size as usize) {
                s += data[i][b * (bin_size as usize) + k];
            }
            let mean = s / bs_f;
            assert!((mean - claimed[i][b]).abs() < 0.001);
        }
    }
}
