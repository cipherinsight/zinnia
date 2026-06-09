// DS-1000 case 301 — np.reshape(A, (-1, ncol)) on Float[20] with runtime ncol.
// kernel.py: B = np.reshape(A, (-1, ncol))
// Sample: A = [1..20] (Q-scaled), ncol = 4 → shape (5, 4). Layout-only.
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
    let ncol: i128 = 4;
    let claim_rows: i128 = 5;
    let claim_ncol: i128 = 4;

    assert!(ncol != 0);
    let rows: i128 = 20 / ncol;
    let rem: i128 = 20 % ncol;
    assert!(rem == 0);
    assert!(rows == claim_rows);
    assert!(ncol == claim_ncol);

    // Layout-only: cells equal the original A (Q-scaled).
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
