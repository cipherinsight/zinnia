// DS-1000 case 419 — SP1 guest.
// Kernel: new_data = data[:, ::-1];
//         bin_data_mean = new_data[:, :(6//bin_size)*bin_size]
//                                .reshape(8, -1, bin_size).mean(axis=-1)[:,::-1]
// Same as 0418 but final result columns reversed.

#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for _ in 0..6 { row.push(sp1_zkvm::io::read::<f64>()); }
        data.push(row);
    }
    let bin_size: u32 = sp1_zkvm::io::read::<u32>();
    let trimmed: usize = ((6u32 / bin_size) * bin_size) as usize;
    let n_bins: usize = trimmed / (bin_size as usize);

    let mut claimed: Vec<Vec<f64>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(n_bins);
        for _ in 0..n_bins { row.push(sp1_zkvm::io::read::<f64>()); }
        claimed.push(row);
    }

    let mut new_data: Vec<Vec<f64>> = Vec::with_capacity(8);
    for i in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(6);
        for j in 0..6 { row.push(data[i][5 - j]); }
        new_data.push(row);
    }

    let bs_f = bin_size as f64;
    let mut tmp: Vec<Vec<f64>> = Vec::with_capacity(8);
    for i in 0..8 {
        let mut row: Vec<f64> = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let mut s: f64 = 0.0;
            for k in 0..(bin_size as usize) {
                s += new_data[i][b * (bin_size as usize) + k];
            }
            row.push(s / bs_f);
        }
        tmp.push(row);
    }

    // Reverse the bin axis of tmp.
    for i in 0..8 {
        for b in 0..n_bins {
            let v = tmp[i][n_bins - 1 - b];
            assert!((v - claimed[i][b]).abs() < 0.001);
        }
    }
}
