// DS-1000 case 417 — SP1 guest.
// Kernel: new_data = data[::-1];
//         bin_data_mean = new_data[:(8//bin_size)*bin_size]
//                                 .reshape(-1, bin_size).mean(axis=1)
// Inputs: data[8] f64, bin_size: u32, claimed[n_bins] f64.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: Vec<f64> = Vec::with_capacity(8);
    for _ in 0..8 {
        data.push(sp1_zkvm::io::read::<f64>());
    }
    let bin_size: u32 = sp1_zkvm::io::read::<u32>();
    let trimmed: usize = ((8u32 / bin_size) * bin_size) as usize;
    let n_bins: usize = trimmed / (bin_size as usize);

    let mut claimed: Vec<f64> = Vec::with_capacity(n_bins);
    for _ in 0..n_bins {
        claimed.push(sp1_zkvm::io::read::<f64>());
    }

    let mut new_data: Vec<f64> = Vec::with_capacity(8);
    for k in 0..8 {
        new_data.push(data[7 - k]);
    }

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
