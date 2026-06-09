// DS-1000 case 454 — `X / np.abs(X).max(axis=1).reshape(-1,1)` on Integer[5,4].
// Quantisation ×2^32 (Q). For each row: linf_q = max(|x_q|);
// result_q = (x_q * Q) / linf_q.

fn abs_i128(v: i128) -> i128 {
    if v < 0 { -v } else { v }
}

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    let x: Array<i128> = array![
         1,  2,  4,  3,
        -2,  4, -1,  0,
         8,  2, -4,  1,
        -3, -5,  0,  1,
         2,  0, -1, -4,
    ];
    // Row abs-max: [4, 4, 8, 5, 4]. claimed_q[k] = (x[k] * Q) / linf[row].
    let claimed: Array<i128> = array![
        Q / 4,         (2 * Q) / 4,   Q,             (3 * Q) / 4,
        (-2 * Q) / 4,  Q,             (-Q) / 4,      0,
        Q,             (2 * Q) / 8,   (-4 * Q) / 8,  Q / 8,
        (-3 * Q) / 5,  -Q,            0,             Q / 5,
        (2 * Q) / 4,   0,             (-Q) / 4,      -Q,
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
        let mut linf: i128 = *abs_fp.at(base);
        for j in 1_u32..4 {
            let cur: i128 = *abs_fp.at(base + j);
            if cur > linf { linf = cur; }
        };
        assert!(linf != 0);
        for j in 0_u32..4 {
            let r: i128 = (*x_fp.at(base + j) * Q) / linf;
            result.append(r);
        };
    };

    let tol: i128 = 1;
    for k in 0_u32..20 {
        let r: i128 = *result.at(k);
        let c: i128 = *claimed.at(k);
        let diff: i128 = if r >= c { r - c } else { c - r };
        assert!(diff <= tol);
    };
}
