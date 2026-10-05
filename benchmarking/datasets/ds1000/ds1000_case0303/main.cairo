// DS-1000 case 303 — B = A[:col].reshape(-1, ncol) where col = (23 // ncol) * ncol.
// kernel.py: col = (A.shape[0] // ncol) * ncol; B = A[:col]; B = np.reshape(B, (-1, ncol)).
// Sample: A = [1..23] (Q-scaled), ncol = 5 → col_keep = 20, rows = 4.
// Layout-only after select; cells with i >= col_keep are zeroed.
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

    // Build expected output cells and assert per-cell.
    let mut k: usize = 0;
    while k < 23 {
        let kk: i128 = k.into();
        let in_range = kk < col_keep;
        let expected: i128 = if in_range { (kk + 1) * Q } else { 0 };
        assert!(*a.at(k) * (if in_range { 1 } else { 0 }) == expected);
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
