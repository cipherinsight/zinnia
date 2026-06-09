// DS-1000 case 414 — SP1 guest program.
// Kernel:
//   bin_data_mean = data[:(data.size // bin_size) * bin_size]
//                       .reshape(-1, bin_size).mean(axis=1)
// Sample: data=[1..8] f64, bin_size=2 → means [1.5, 3.5, 5.5, 7.5].

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    const N: usize = 8;
    let mut data: [f64; N] = [0.0; N];
    for k in 0..N { data[k] = sp1_zkvm::io::read::<f64>(); }
    let bin_size: u32 = sp1_zkvm::io::read::<u32>();
    let bs = bin_size as usize;
    let trimmed = (N / bs) * bs;
    let n_bins = trimmed / bs;

    let mut claimed: Vec<f64> = Vec::with_capacity(n_bins);
    for _ in 0..n_bins { claimed.push(sp1_zkvm::io::read::<f64>()); }

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
