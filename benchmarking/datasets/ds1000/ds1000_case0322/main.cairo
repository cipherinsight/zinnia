// DS-1000 case 322 — Cairo baseline.
// Kernel: `np.argwhere(a == np.min(a))` on Integer[2, 2].
// Sample input has a unique minimum so the result is one (row, col) pair.

#[executable]
pub fn main() {
    let mut data: Array<i128> = ArrayTrait::new();
    data.append(5);
    data.append(9);
    data.append(1);
    data.append(7);

    let claimed_row: u128 = 1;
    let claimed_col: u128 = 0;

    let mut min_val: i128 = *data.at(0);
    for i in 0_u32..2 {
        for j in 0_u32..2 {
            let v = *data.at(i * 2 + j);
            if v < min_val {
                min_val = v;
            }
        };
    };

    let mut out_row: u128 = 0;
    let mut out_col: u128 = 0;
    let mut found: bool = false;
    for i in 0_u32..2 {
        for j in 0_u32..2 {
            let v = *data.at(i * 2 + j);
            if !found && v == min_val {
                out_row = i.into();
                out_col = j.into();
                found = true;
            }
        };
    };
    assert!(found);

    assert!(out_row == claimed_row);
    assert!(out_col == claimed_col);
}
