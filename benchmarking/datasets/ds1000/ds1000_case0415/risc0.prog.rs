// DS-1000 case 415 — Risc0 guest. 1-D bin max.

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

    for b in 0..n_bins {
        let mut m: f64 = f64::NEG_INFINITY;
        for k in 0..bs {
            let v = data[b * bs + k];
            if v > m { m = v; }
        }
        assert!((m - claimed[b]).abs() < 0.001);
    }
}
