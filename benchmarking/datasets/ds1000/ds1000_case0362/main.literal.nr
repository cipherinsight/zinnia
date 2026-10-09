// DS-1000 case 0362, literal translation of kernel.py (LITERAL_BASELINES.md).
// Constant del_col, mask and filtering fold to deletion of column zero (L3).
// @outputs: float[40]
fn main(a: [Field; 48]) -> pub [Field; 40] {
    let del_col = [12, 14, 1, 9];
    let mut mask = [false; 4];
    for i in 0..4 { mask[i] = del_col[i] <= 6; }
    // del_col[mask] - 1 = [0].
    let mut result = [0; 40];
    for i in 0..8 { for j in 1..6 {
        result[i*5+j-1] = noir_hw::checked(a[i*6+j]);
    }}
    result
}
