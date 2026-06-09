// DS-1000 case 437 — Cairo baseline.
// Kernel: mask = (a.min(axis=1, keepdims=1) == a) on Float[10, 5].
// Sample: a[i][j] = (i+1)*10 + (j+1)*0.1, with a[i][2] = (i+1)*10 - 99 so the
// per-row min sits at column 2. Some values negative → i128.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;

    let mut a: Array<i128> = ArrayTrait::new();
    for i in 0_i128..10 {
        for j in 0_i128..5 {
            let q_v: i128 = if j == 2 {
                ((i + 1) * 10 - 99) * Q
            } else {
                let int_part: i128 = (i + 1) * 10;
                let frac_num: i128 = j + 1;
                int_part * Q + (frac_num * Q) / 10
            };
            a.append(q_v);
        };
    };

    let mut expected: Array<i128> = ArrayTrait::new();
    for i in 0_u32..10 {
        for j in 0_u32..5 {
            expected.append(if j == 2 { 1 } else { 0 });
        };
    };

    for i in 0_u32..10 {
        let mut row_min: i128 = *a.at((i * 5).try_into().unwrap());
        for j in 1_u32..5 {
            let idx: u32 = i * 5 + j;
            let cur: i128 = *a.at(idx.try_into().unwrap());
            if cur < row_min { row_min = cur; };
        };
        for j in 0_u32..5 {
            let idx: u32 = i * 5 + j;
            let v: i128 = *a.at(idx.try_into().unwrap());
            let eq: i128 = if v == row_min { 1 } else { 0 };
            assert!(eq == *expected.at(idx.try_into().unwrap()));
        };
    };
}
