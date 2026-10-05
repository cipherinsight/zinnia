// DS-1000 case 435 — Cairo baseline.
// Kernel: a[1, :] = 0; a[:, 0] = 0 on Integer[4, 4]. Sample: a[i][j] = i*4+j+1.

#[executable]
pub fn main() {
    let mut a: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            a.append(i * 4 + j + 1);
        };
    };

    let mut expected: Array<i128> = ArrayTrait::new();
    for i in 0_i128..4 {
        for j in 0_i128..4 {
            let idx: u32 = (i * 4 + j).try_into().unwrap();
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if i == 1 || j == 0 { 0 } else { v };
            expected.append(out);
        };
    };

    for i in 0_u32..4 {
        for j in 0_u32..4 {
            let idx: u32 = i * 4 + j;
            let v: i128 = *a.at(idx.try_into().unwrap());
            let out: i128 = if i == 1 || j == 0 { 0 } else { v };
            assert!(out == *expected.at(idx.try_into().unwrap()));
        };
    };
}
