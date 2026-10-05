// DS-1000 case 414 — Risc0 guest. 1-D bin mean.

use risc0_zkvm::guest::env;

fn main() {
    const N: usize = 8;
    let mut data: [f64; N] = [0.0; N];
    for k in 0..N { data[k] = env::read(); }
    let bin_size: u32 = env::read();
    let bs = bin_size as usize;
    let trimmed = (N / bs) * bs;
    let n_bins = trimmed / bs;

    let mut claimed: Vec<f64> = Vec::with_capacity(n_bins);
    for _ in 0..n_bins { claimed.push(env::read()); }

    let bs_f = bin_size as f64;
    for b in 0..n_bins {
        let mut s: f64 = 0.0;
        for k in 0..bs {
            s += data[b * bs + k];
        }
        let mean = s / bs_f;
        assert!((mean - claimed[b]).abs() < 0.001);
    }
}
