// DS-1000 case 415 — SP1 guest program.
// Kernel:
//   bin_data_max = data[:(data.size // bin_size) * bin_size]
//                      .reshape(-1, bin_size).max(axis=1)
// Sample: data=[1,2,4,3,6,5,7,8] f64, bin_size=2 → maxes [2, 4, 6, 8].
// (Maximum is in a non-first position in two of the four bins.)

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

    for b in 0..n_bins {
        let mut m: f64 = f64::NEG_INFINITY;
        for k in 0..bs {
            let v = data[b * bs + k];
            if v > m { m = v; }
        }
        assert!((m - claimed[b]).abs() < 0.001);
    }
}
