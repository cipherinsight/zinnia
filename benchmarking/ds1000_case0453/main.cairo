// DS-1000 case 453 — `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1,1)` on
// Integer[5,4]. Pythagorean rows: ||row||₂ = 5 for every row, so result_q[i][j]
// = (x_q[i][j] * Q) / l2_q, with l2_q = 5 * Q.
// Quantisation ×2^32 (Q). Sqrt via Newton iteration on `target = ss_q * Q`.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    const QU: u128 = 4_294_967_296;

    let x: Array<i128> = array![
        3, 4, 0, 0,
        0, 3, 4, 0,
        0, 0, 3, 4,
        4, 0, 0, 3,
        3, 0, 4, 0,
    ];
    // claimed: integer values pre-divided by l2=5 then ×Q.
    let claimed: Array<i128> = array![
        (3 * Q) / 5, (4 * Q) / 5, 0,           0,
        0,           (3 * Q) / 5, (4 * Q) / 5, 0,
        0,           0,           (3 * Q) / 5, (4 * Q) / 5,
        (4 * Q) / 5, 0,           0,           (3 * Q) / 5,
        (3 * Q) / 5, 0,           (4 * Q) / 5, 0,
    ];

    let mut x_fp: Array<i128> = ArrayTrait::new();
    for k in 0_u32..20 {
        x_fp.append(*x.at(k) * Q);
    };

    let mut result: Array<i128> = ArrayTrait::new();
    for i in 0_u32..5 {
        let base: u32 = i * 4;
        // sum of squares in Q-scale: ss_q = sum((x_q * x_q) / Q).
        let mut ss_q: i128 = 0;
        for j in 0_u32..4 {
            let v: i128 = *x_fp.at(base + j);
            let sq: i128 = (v * v) / Q;
            ss_q = ss_q + sq;
        };
        assert!(ss_q >= 0);
        // l2_q = sqrt(ss_q · Q) via Newton iteration in u128 domain.
        let target: u128 = (ss_q.try_into().unwrap()) * QU;
        let mut g: u128 = if target > 0 { target } else { 1 };
        let mut prev: u128 = 0;
        while g != prev {
            prev = g;
            g = (g + target / g) / 2;
        };
        let l2_q: i128 = g.try_into().unwrap();
        assert!(l2_q != 0);
        for j in 0_u32..4 {
            let r: i128 = (*x_fp.at(base + j) * Q) / l2_q;
            result.append(r);
        };
    };

    // Tolerance ±2 in Q-scale to absorb sqrt-truncation and qdiv rounding.
    let tol: i128 = 2;
    for k in 0_u32..20 {
        let r: i128 = *result.at(k);
        let c: i128 = *claimed.at(k);
        let diff: i128 = if r >= c { r - c } else { c - r };
        assert!(diff <= tol);
    };
}
