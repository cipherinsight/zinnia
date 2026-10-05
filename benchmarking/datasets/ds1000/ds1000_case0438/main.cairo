// DS-1000 case 438 — Cairo baseline.
// Kernel: result = np.corrcoef(post, distance)[0][1] on Integer[4] × Integer[4].
// Sample: post = [1, 2, 3, 4]; distance = [2, 4, 6, 8] → perfect linear, r=1.0.
// Quantisation ×2^32. cov/(σ_x·σ_y) with witness sqrt via Newton.

#[executable]
pub fn main() {
    const Q: i128 = 4_294_967_296;
    const QU: u128 = 4_294_967_296;

    let post_int: Array<i64> = array![1, 2, 3, 4];
    let dist_int: Array<i64> = array![2, 4, 6, 8];
    // FloatCast each to Q-scale.
    let mut xs: Array<i128> = ArrayTrait::new();
    let mut ys: Array<i128> = ArrayTrait::new();
    for k in 0_u32..4 {
        let p: i64 = *post_int.at(k.try_into().unwrap());
        let d: i64 = *dist_int.at(k.try_into().unwrap());
        let p_i: i128 = p.into();
        let d_i: i128 = d.into();
        xs.append(p_i * Q);
        ys.append(d_i * Q);
    };

    let n_q: i128 = 4 * Q;
    let mut xsum: i128 = 0;
    let mut ysum: i128 = 0;
    for k in 0_u32..4 {
        xsum = xsum + *xs.at(k.try_into().unwrap());
        ysum = ysum + *ys.at(k.try_into().unwrap());
    };
    let xbar: i128 = (xsum * Q) / n_q;
    let ybar: i128 = (ysum * Q) / n_q;

    let mut cov: i128 = 0;
    let mut vx: i128 = 0;
    let mut vy: i128 = 0;
    for k in 0_u32..4 {
        let xk: i128 = *xs.at(k.try_into().unwrap());
        let yk: i128 = *ys.at(k.try_into().unwrap());
        let dx: i128 = xk - xbar;
        let dy: i128 = yk - ybar;
        cov = cov + (dx * dy) / Q;
        vx = vx + (dx * dx) / Q;
        vy = vy + (dy * dy) / Q;
    };

    // prod_v = qmul(vx, vy); denom = sqrt(prod_v).
    let prod_v: i128 = (vx * vy) / Q;
    let target: u128 = (prod_v.try_into().unwrap()) * QU;
    let mut g: u128 = if target > 0 { target } else { 1 };
    let mut prev: u128 = 0;
    while g != prev {
        prev = g;
        g = (g + target / g) / 2;
    };
    let denom: i128 = g.try_into().unwrap();

    // r = qdiv(cov, denom) = (cov * Q) / denom.
    let r: i128 = (cov * Q) / denom;

    let claimed_r: i128 = Q; // 1.0
    let tol: i128 = (5 * Q) / 1000;

    assert!(r + tol >= claimed_r);
    assert!(r <= claimed_r + tol);
}
