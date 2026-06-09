// DS-1000 case 452 — `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on Integer[5,4].
// Quantisation ×2^32 (Q). For each row: l1_q = sum(|x_q|);
// result_q = (x_q * Q) / l1_q (rescale to stay in Q-scale).

fn abs_i128(v: i128) -> i128 {
    if v < 0 { -v } else { v }
}

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let x: Array<i128> = array![
         1,  2,  3,  4,
        -1, -2, -3,  4,
         5,  2, -2,  1,
        -3, -3, -2, -2,
         4, -3,  2,  1,
    ];

    // Pre-compute claimed: result_q[i][j] = (x[i][j] * Q) / l1[i]
    // where l1[i] = sum of |x[i][j]|. For row 0: 10. Row 1: 10. Row 2: 10.
    // Row 3: 10. Row 4: 10. So all rows divide by 10.
    let claimed: Array<i128> = array![
        Q / 10,  (2 * Q) / 10,  (3 * Q) / 10,  (4 * Q) / 10,
        -Q / 10, (-2 * Q) / 10, (-3 * Q) / 10, (4 * Q) / 10,
        (5 * Q) / 10, (2 * Q) / 10, (-2 * Q) / 10, Q / 10,
        (-3 * Q) / 10, (-3 * Q) / 10, (-2 * Q) / 10, (-2 * Q) / 10,
        (4 * Q) / 10, (-3 * Q) / 10, (2 * Q) / 10, Q / 10,
    ];

    let mut x_fp: Array<i128> = ArrayTrait::new();
    let mut abs_fp: Array<i128> = ArrayTrait::new();
    for k in 0_u32..20 {
        let v: i128 = *x.at(k) * Q;
        x_fp.append(v);
        abs_fp.append(abs_i128(v));
    };

    let mut result: Array<i128> = ArrayTrait::new();
    for i in 0_u32..5 {
        let base: u32 = i * 4;
        let mut l1: i128 = 0;
        for j in 0_u32..4 {
            l1 = l1 + *abs_fp.at(base + j);
        };
        assert!(l1 != 0);
        for j in 0_u32..4 {
            let r: i128 = (*x_fp.at(base + j) * Q) / l1;
            result.append(r);
        };
    };

    // Tolerance ±1 in Q-scale to accommodate truncation between (Q/10) literals
    // and the divided values. Equivalent to ~2.3e-10 absolute error.
    let tol: i128 = 1;
    for k in 0_u32..20 {
        let r: i128 = *result.at(k);
        let c: i128 = *claimed.at(k);
        let diff: i128 = if r >= c { r - c } else { c - r };
        assert!(diff <= tol);
    };
}
