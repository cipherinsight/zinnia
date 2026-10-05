// DS-1000 case 320 — Cairo baseline.
// Kernel: `np.unravel_index(a.argmin(), a.shape)` on Float[5, 6] → (row, col).
// Quantisation ×2^32 (Q) to match Zinnia's `precision_bits = 32`. argmin is
// monotonic so the index is identical regardless of scale; signed i128 values
// are used because the minimum cell is negative.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    // a[i][j] = ((i+1)*10 + (j+1)) / 100, except a[2][4] = -99.0 (the min).
    let mut data: Array<i128> = ArrayTrait::new();
    for i in 0_u32..5 {
        for j in 0_u32..6 {
            let q_v: i128 = if i == 2 && j == 4 {
                -99_i128 * Q
            } else {
                let num: i128 = ((i + 1) * 10 + (j + 1)).into();
                (num * Q) / 100
            };
            data.append(q_v);
        };
    };
    let claimed_row: u128 = 2;
    let claimed_col: u128 = 4;

    let mut best_val: i128 = *data.at(0);
    let mut best_idx: u128 = 0;
    for k in 1_u32..30 {
        let v = *data.at(k);
        if v < best_val {
            best_val = v;
            best_idx = k.into();
        }
    };
    let row = best_idx / 6;
    let col = best_idx % 6;

    assert!(row == claimed_row);
    assert!(col == claimed_col);
}
