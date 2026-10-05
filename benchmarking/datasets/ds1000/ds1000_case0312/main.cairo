// DS-1000 case 312 — np.unravel_index(a.argmax(), a.shape) on Float[8, 6].
// C-order unravel: row = idx / 6, col = idx % 6.
// Sample: a[3][2] = 99.0 → flat 20 → (row=3, col=2).
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let mut data: Array<u128> = ArrayTrait::new();
    for i in 0_u32..8 {
        for j in 0_u32..6 {
            let q_v: u128 = if i == 3 && j == 2 {
                99_u128 * Q
            } else {
                let num: u128 = ((i + 1) * 10 + (j + 1)).into();
                (num * Q) / 100
            };
            data.append(q_v);
        };
    };
    let claim_row: u128 = 3;
    let claim_col: u128 = 2;

    let mut best_val: u128 = *data.at(0);
    let mut best_idx: u128 = 0;
    let mut k: u32 = 1;
    while k < 48 {
        let v: u128 = *data.at(k.try_into().unwrap());
        if v > best_val {
            best_val = v;
            best_idx = k.into();
        }
        k += 1;
    };
    let row: u128 = best_idx / 6;
    let col: u128 = best_idx % 6;

    assert!(row == claim_row);
    assert!(col == claim_col);
}
