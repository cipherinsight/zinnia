// DS-1000 case 433 — Cairo baseline.
// Kernel: a[zero_rows, :] = 0; a[:, zero_cols] = 0 on Integer[4, 4] with
// scalar zero_rows, zero_cols (runtime). Sample: a[i][j] = i*4 + j + 1;
// zero_rows=2, zero_cols=1.
// Integer values — no Q scaling needed (still consistent with i128 storage).

#[executable]
pub fn main() {
    let mut a: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            a.append(i * 4 + j + 1);
        };
    };
    let zero_rows: i128 = 2;
    let zero_cols: i128 = 1;

    // Build expected.
    let mut expected: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            let row_match: bool = i == zero_rows;
            let col_match: bool = j == zero_cols;
            let idx: u32 = (i * 4 + j).try_into().unwrap();
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if row_match || col_match { 0 } else { v };
            expected.append(out);
        };
    };

    for i in 0_u32..4 {
        let i_i: i128 = i.into();
        let row_match: bool = i_i == zero_rows;
        for j in 0_u32..4 {
            let j_i: i128 = j.into();
            let col_match: bool = j_i == zero_cols;
            let any: bool = row_match || col_match;
            let idx: u32 = i * 4 + j;
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if any { 0 } else { v };
            assert!(out == *expected.at(idx.try_into().unwrap()));
        };
    };
}
