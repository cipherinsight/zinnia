// DS-1000 case 314 — first argmax (unravel), set that cell to a.min(), then argmax again.
// Float[8, 6]. Cairo Array<T> is append-only, so we materialise a fresh array on the
// replace step. Quantisation ×2^32 (Q).

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
    let claim_row: u128 = 7;
    let claim_col: u128 = 5;

    // First argmax (C-order flat).
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

    // a.min().
    let mut min_val: u128 = *data.at(0);
    let mut k2: u32 = 1;
    while k2 < 48 {
        let v: u128 = *data.at(k2.try_into().unwrap());
        if v < min_val {
            min_val = v;
        }
        k2 += 1;
    };

    // Build modified array: replace best_idx with min_val.
    let mut data2: Array<u128> = ArrayTrait::new();
    let mut k3: u32 = 0;
    while k3 < 48 {
        let k3_u128: u128 = k3.into();
        let v: u128 = if k3_u128 == best_idx {
            min_val
        } else {
            *data.at(k3.try_into().unwrap())
        };
        data2.append(v);
        k3 += 1;
    };

    // Second argmax → unravel (C-order).
    let mut best_val2: u128 = *data2.at(0);
    let mut best_idx2: u128 = 0;
    let mut k4: u32 = 1;
    while k4 < 48 {
        let v: u128 = *data2.at(k4.try_into().unwrap());
        if v > best_val2 {
            best_val2 = v;
            best_idx2 = k4.into();
        }
        k4 += 1;
    };
    let row: u128 = best_idx2 / 6;
    let col: u128 = best_idx2 % 6;

    assert!(row == claim_row);
    assert!(col == claim_col);
}
