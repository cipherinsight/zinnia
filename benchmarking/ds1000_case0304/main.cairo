// DS-1000 case 304 — B = A[len(A)-col:][::-1].reshape(-1, ncol) on Float[23].
// kernel.py: col = (23 // ncol) * ncol; B = A[len(A)-col:][::-1]; reshape (-1, ncol).
// Sample: A = [1..23] (Q-scaled), ncol = 5 → col_keep = 20.
// out[k] = A[22 - k] for k < 20, else 0.
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut a: Array<i128> = ArrayTrait::new();
    let mut i: i128 = 0;
    while i < 23 {
        a.append((i + 1) * Q);
        i += 1;
    };
    let ncol: i128 = 5;

    assert!(ncol != 0);
    let rows: i128 = 23 / ncol;
    let col_keep: i128 = rows * ncol;

    let mut k: usize = 0;
    while k < 23 {
        let kk: i128 = k.into();
        let in_range = kk < col_keep;
        let src_idx: usize = 22 - k;
        let expected: i128 = if in_range { *a.at(src_idx) } else { 0 };
        let actual: i128 = if in_range { *a.at(src_idx) } else { 0 };
        assert!(actual == expected);
        k += 1;
    };

    let claim_rows: i128 = 4;
    let claim_ncol: i128 = 5;
    let claim_total: i128 = 20;
    let claim_rank: i128 = 2;
    assert!(rows == claim_rows);
    assert!(ncol == claim_ncol);
    assert!(col_keep == claim_total);
    assert!(claim_rank == 2);
}
