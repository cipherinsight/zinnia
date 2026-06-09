// DS-1000 case 460 — `np.mean(NA.astype(float), axis=0)` over Integer[5].
// A: Float[5] witnessed but unused. NA = [2,4,6,8,10]; mean = 6.
// Quantisation ×2^32 (Q).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // A: 0.1, 0.2, 0.3, 0.4, 0.5 — quantised but unused.
    let mut a: Array<u128> = ArrayTrait::new();
    a.append(Q / 10);
    a.append((2 * Q) / 10);
    a.append((3 * Q) / 10);
    a.append((4 * Q) / 10);
    a.append((5 * Q) / 10);
    let _ = a.len();

    let na: Array<u128> = array![2, 4, 6, 8, 10];
    let claimed: u128 = 6 * Q;

    // Cast NA to fp (× Q).
    let mut na_fp: Array<u128> = ArrayTrait::new();
    for k in 0_u32..5 {
        na_fp.append(*na.at(k) * Q);
    };

    let mut sum: u128 = *na_fp.at(0);
    for k in 1_u32..5 {
        sum = sum + *na_fp.at(k);
    };
    let n: u128 = 5 * Q;
    // qdiv: (sum * Q) / n.
    let mean: u128 = (sum * Q) / n;

    assert!(mean == claimed);
}
