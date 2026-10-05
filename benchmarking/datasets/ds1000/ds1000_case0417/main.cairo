// DS-1000 case 417 — Cairo baseline.
// Kernel: new_data = data[::-1];
//         bin_data_mean = new_data[:(8//bin_size)*bin_size]
//                                 .reshape(-1, bin_size).mean(axis=1)
// Sample: data = [1..8]; bin_size=2 → means [7.5, 5.5, 3.5, 1.5].
// Quantisation ×2^32 to match Zinnia.

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

    // Reverse.
    let mut new_data: Array<u128> = ArrayTrait::new();
    for k in 0_u32..8 {
        let idx: u32 = 7 - k;
        new_data.append(*data.at(idx.try_into().unwrap()));
    };

    // Per-bin mean. Pairs: (8+7)/2=7.5, (6+5)/2=5.5, (4+3)/2=3.5, (2+1)/2=1.5.
    let mut expected: Array<u128> = ArrayTrait::new();
    // 7.5*Q = 7*Q + Q/2, etc.
    expected.append(7 * Q + Q / 2);
    expected.append(5 * Q + Q / 2);
    expected.append(3 * Q + Q / 2);
    expected.append(1 * Q + Q / 2);

    let bs_q: u128 = (bin_size).into();
    for b in 0_u32..n_bins {
        let mut s: u128 = 0;
        for k in 0_u32..bin_size {
            let idx: u32 = b * bin_size + k;
            s = s + *new_data.at(idx.try_into().unwrap());
        };
        let mean_q: u128 = s / bs_q;
        assert!(mean_q == *expected.at(b.try_into().unwrap()));
    };
}
