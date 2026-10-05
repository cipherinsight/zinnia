// DS-1000 case 419 — Cairo baseline.
// Kernel: new_data = data[:, ::-1];
//         bin_data_mean = new_data[:, :(6//bin_size)*bin_size]
//                                .reshape(8, -1, bin_size).mean(axis=-1)[:,::-1]
// Same as 0418 but the result bin axis is reversed back.
// Quantisation ×2^32.

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

    let mut new_data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let src: u32 = i * 6 + (5 - j);
            new_data.append(*data.at(src.try_into().unwrap()));
        };
    };

    // tmp = per-row binned means.
    let bs_q: u128 = (bin_size).into();
    let mut tmp: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for b in 0_u32..n_bins {
            let mut s: u128 = 0;
            for k in 0_u32..bin_size {
                let idx: u32 = i * 6 + b * bin_size + k;
                s = s + *new_data.at(idx.try_into().unwrap());
            };
            tmp.append(s / bs_q);
        };
    };

    // Expected after final reverse: same as 0416 (base+1.5, base+3.5, base+5.5).
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        let base: u128 = ((i + 1) * 10).into();
        expected.append(base * Q + 1 * Q + Q / 2);
        expected.append(base * Q + 3 * Q + Q / 2);
        expected.append(base * Q + 5 * Q + Q / 2);
    };

    // Reverse bin axis of tmp and compare.
    for i in 0_u32..8 {
        for b in 0_u32..n_bins {
            let src: u32 = i * n_bins + (n_bins - 1 - b);
            let eidx: u32 = i * n_bins + b;
            assert!(*tmp.at(src.try_into().unwrap()) == *expected.at(eidx.try_into().unwrap()));
        };
    };
}
