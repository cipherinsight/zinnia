// DS-1000 case 434 — Cairo baseline.
// Kernel: a[zero_rows, :] = 0; a[:, zero_cols] = 0 on Integer[4, 4]
// with zero_rows: Integer[2], zero_cols: Integer[2].
// Sample: a[i][j] = i*4+j+1; zero_rows = [0, 2]; zero_cols = [1, 3].

#[executable]
pub fn main() {
    let mut a: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            a.append(i * 4 + j + 1);
        };
    };
    let zero_rows: Array<i128> = array![0, 2];
    let zero_cols: Array<i128> = array![1, 3];

    // Build expected.
    let mut expected: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            let rz: bool = i == *zero_rows.at(0) || i == *zero_rows.at(1);
            let cz: bool = j == *zero_cols.at(0) || j == *zero_cols.at(1);
            let idx: u32 = (i * 4 + j).try_into().unwrap();
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if rz || cz { 0 } else { v };
            expected.append(out);
        };
    };

    for i in 0_u32..4 {
        let i_i: i128 = i.into();
        let rz: bool = i_i == *zero_rows.at(0) || i_i == *zero_rows.at(1);
        for j in 0_u32..4 {
            let j_i: i128 = j.into();
            let cz: bool = j_i == *zero_cols.at(0) || j_i == *zero_cols.at(1);
            let any: bool = rz || cz;
            let idx: u32 = i * 4 + j;
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if any { 0 } else { v };
            assert!(out == *expected.at(idx.try_into().unwrap()));
        };
    };
}
