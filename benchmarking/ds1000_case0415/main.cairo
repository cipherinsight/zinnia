// DS-1000 case 415 — Cairo baseline.
// Kernel:
//   bin_data_max = data[:(8 // bin_size) * bin_size]
//                      .reshape(-1, bin_size).max(axis=1)
// Sample: data=[1,2,4,3,6,5,7,8]; bin_size=2 → maxes [2, 4, 6, 8].
// Quantisation ×2^32.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let bin_size: u32 = 2;
    let trimmed: u32 = (8 / bin_size) * bin_size;
    let n_bins: u32 = trimmed / bin_size;

    let mut data: Array<u128> = ArrayTrait::new();
    data.append(1 * Q);
    data.append(2 * Q);
    data.append(4 * Q);
    data.append(3 * Q);
    data.append(6 * Q);
    data.append(5 * Q);
    data.append(7 * Q);
    data.append(8 * Q);

    let mut expected: Array<u128> = ArrayTrait::new();
    expected.append(2 * Q);
    expected.append(4 * Q);
    expected.append(6 * Q);
    expected.append(8 * Q);

    for b in 0_u32..n_bins {
        let mut m: u128 = 0;
        for k in 0_u32..bin_size {
            let idx: u32 = b * bin_size + k;
            let v: u128 = *data.at(idx.try_into().unwrap());
            if v > m { m = v; }
        };
        assert!(m == *expected.at(b.try_into().unwrap()));
    };
}
