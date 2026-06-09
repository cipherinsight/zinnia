// DS-1000 case 414 — Cairo baseline.
// Kernel:
//   bin_data_mean = data[:(8 // bin_size) * bin_size]
//                       .reshape(-1, bin_size).mean(axis=1)
// Sample: data=[1..8]; bin_size=2 → means [1.5, 3.5, 5.5, 7.5].
// Quantisation ×2^32.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let bin_size: u32 = 2;
    let trimmed: u32 = (8 / bin_size) * bin_size;
    let n_bins: u32 = trimmed / bin_size;

    let mut data: Array<u128> = ArrayTrait::new();
    for i in 1_u128..9 {
        data.append(i * Q);
    };

    // expected per bin: (j0 + j1)/2 with j0=2b+1, j1=2b+2 → integer base + 0.5.
    let mut expected: Array<u128> = ArrayTrait::new();
    expected.append(1 * Q + Q / 2);
    expected.append(3 * Q + Q / 2);
    expected.append(5 * Q + Q / 2);
    expected.append(7 * Q + Q / 2);

    let bs_q: u128 = (bin_size).into();
    for b in 0_u32..n_bins {
        let mut s: u128 = 0;
        for k in 0_u32..bin_size {
            let idx: u32 = b * bin_size + k;
            s = s + *data.at(idx.try_into().unwrap());
        };
        let mean_q: u128 = s / bs_q;
        assert!(mean_q == *expected.at(b.try_into().unwrap()));
    };
}
