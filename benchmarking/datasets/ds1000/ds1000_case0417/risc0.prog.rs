// DS-1000 case 417 — Risc0 guest.

use risc0_zkvm::guest::env;

fn main() {
    let mut data: Vec<f64> = Vec::with_capacity(8);
    for _ in 0..8 { data.push(env::read()); }
    let bin_size: u32 = env::read();
    let trimmed: usize = ((8u32 / bin_size) * bin_size) as usize;
    let n_bins: usize = trimmed / (bin_size as usize);

    let mut claimed: Vec<f64> = Vec::with_capacity(n_bins);
    for _ in 0..n_bins { claimed.push(env::read()); }

    let mut new_data: Vec<f64> = Vec::with_capacity(8);
    for k in 0..8 { new_data.push(data[7 - k]); }

    let bs_f = bin_size as f64;
    for b in 0..n_bins {
        let mut s: f64 = 0.0;
        for k in 0..(bin_size as usize) {
            s += new_data[b * (bin_size as usize) + k];
        }
        let mean = s / bs_f;
        assert!((mean - claimed[b]).abs() < 0.001);
    }
}
