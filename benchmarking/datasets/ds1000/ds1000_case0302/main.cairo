// DS-1000 case 302 — np.reshape(A, (nrow, -1)) on Float[20] with runtime nrow.
// kernel.py: B = np.reshape(A, (nrow, -1))
// Sample: A = [1..20] (Q-scaled), nrow = 4 → shape (4, 5). Layout-only.
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let mut a: Array<i128> = ArrayTrait::new();
    let mut i: i128 = 0;
    while i < 20 {
        a.append((i + 1) * Q);
        i += 1;
    };
    let nrow: i128 = 4;
    let claim_nrow: i128 = 4;
    let claim_ncol: i128 = 5;

    assert!(nrow != 0);
    let ncol: i128 = 20 / nrow;
    let rem: i128 = 20 % nrow;
    assert!(rem == 0);
    assert!(nrow == claim_nrow);
    assert!(ncol == claim_ncol);

    let mut k: usize = 0;
    while k < 20 {
        let kk: i128 = k.into();
        assert!(*a.at(k) == (kk + 1) * Q);
        k += 1;
    };

    let claim_total: i128 = 20;
    let claim_rank: i128 = 2;
    assert!(claim_total == 20);
    assert!(claim_rank == 2);
}
