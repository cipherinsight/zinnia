// DS-1000 case 478 — Cairo baseline.
// einsum('ii->i', a) extracts the diagonal of a 4×4 int matrix; subsequent
// in-place zeroing of `a` and writing back into `result` are no-ops semantically.

#[executable]
pub fn main() {
    // a[i][j] = i*4 + j + 1 (row-major).
    let mut a: Array<i128> = ArrayTrait::new();
    for v in 1_i128..17 {
        a.append(v);
    };
    let claimed: Array<i128> = array![1, 6, 11, 16];

    let mut result: Array<i128> = ArrayTrait::new();
    for i in 0_u32..4 {
        let idx: u32 = i * 4 + i;
        result.append(*a.at(idx));
    };
    // save = result.copy()
    let mut save: Array<i128> = ArrayTrait::new();
    for i in 0_u32..4 {
        save.append(*result.at(i));
    };
    // a[...] = 0
    let mut a2: Array<i128> = ArrayTrait::new();
    for _ in 0_u32..16 {
        a2.append(0);
    };
    // result[...] = save (rebuild result from save).
    let mut result2: Array<i128> = ArrayTrait::new();
    for i in 0_u32..4 {
        result2.append(*save.at(i));
    };

    for k in 0_u32..4 {
        assert!(*result2.at(k) == *claimed.at(k));
    };
}
