// DS-1000 case 436 — Cairo baseline.
// Kernel: mask = (a.max(axis=1, keepdims=1) == a) on Float[10, 5].
// Sample: a[i][j] = (i+1)*10 + (j+1)*0.1, with a[i][2] bumped to (i+1)*10+99
// so per-row max sits at column 2. Output Bool[10, 5].
// Quantisation ×2^32. All sample values are non-negative.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u32..10 {
        for j in 0_u32..5 {
            let q_v: u128 = if j == 2 {
                let base: u128 = ((i + 1) * 10).into();
                base * Q + 99 * Q
            } else {
                let int_part: u128 = ((i + 1) * 10).into();
                let frac_num: u128 = ((j + 1)).into();
                int_part * Q + (frac_num * Q) / 10
            };
            a.append(q_v);
        };
    };

    // Expected: 1 at column 2 per row, 0 elsewhere.
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u32..10 {
        for j in 0_u32..5 {
            expected.append(if j == 2 { 1 } else { 0 });
        };
    };

    for i in 0_u32..10 {
        let mut row_max: u128 = *a.at((i * 5).try_into().unwrap());
        for j in 1_u32..5 {
            let idx: u32 = i * 5 + j;
            let cur: u128 = *a.at(idx.try_into().unwrap());
            if cur > row_max { row_max = cur; };
        };
        for j in 0_u32..5 {
            let idx: u32 = i * 5 + j;
            let v: u128 = *a.at(idx.try_into().unwrap());
            // exact equality on Q-scaled integers.
            let eq: u128 = if v == row_max { 1 } else { 0 };
            assert!(eq == *expected.at(idx.try_into().unwrap()));
        };
    };
}
