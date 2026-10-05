// DS-1000 case 416 — Cairo baseline.
// Kernel: bin_data_mean = data[:, :(6//bin_size)*bin_size]
//                              .reshape(8, -1, bin_size).mean(axis=-1)
// Sample: data[i][j] = (i+1)*10 + (j+1); bin_size=2. Per-bin mean over two
// consecutive integers (j_lo + j_lo+1)/2 (an exact half-integer).
// Quantisation ×2^32 to match Zinnia.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let bin_size: u32 = 2;
    let trimmed: u32 = (6 / bin_size) * bin_size;
    let n_bins: u32 = trimmed / bin_size;

    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let v: u128 = ((i + 1) * 10 + (j + 1)).into();
            data.append(v * Q);
        };
    };

    // expected[i][b] = (j0 + j1) / 2 with j0=2b, j1=2b+1 → integer base + 0.5.
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for b in 0_u32..n_bins {
            let j0: u32 = b * bin_size;
            let j1: u32 = b * bin_size + 1;
            let sum_u: u128 = ((i + 1) * 10 + (j0 + 1)).into();
            let sum_u2: u128 = ((i + 1) * 10 + (j1 + 1)).into();
            // mean in Q-scale: ((sum)*Q)/2
            expected.append(((sum_u + sum_u2) * Q) / 2);
        };
    };

    // Recompute mean per bin and compare exactly.
    let bs_q: u128 = (bin_size).into();
    for i in 0_u32..8 {
        for b in 0_u32..n_bins {
            let mut s: u128 = 0;
            for k in 0_u32..bin_size {
                let idx: u32 = i * 6 + b * bin_size + k;
                s = s + *data.at(idx.try_into().unwrap());
            };
            let mean_q: u128 = s / bs_q;
            let eidx: u32 = i * n_bins + b;
            assert!(mean_q == *expected.at(eidx.try_into().unwrap()));
        };
    };
}
