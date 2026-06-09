// DS-1000 case 439 — `X.T[:, :, None] * X.T[:, None]` on Float[10,20].
// Result: Float[20, 10, 10] where result[k][i][j] = X[i][k] * X[j][k].
// X[i][k] = 0.01 * (i+1) + 0.001 * (k+1).
// Quantisation ×2^32 (Q). qmul = (a*b) / Q.

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut x: Array<u128> = ArrayTrait::new();  // row-major [10][20].
    for i in 0_u32..10 {
        for k in 0_u32..20 {
            let num: u128 = ((i + 1) * 10 + (k + 1)).into();
            x.append((num * Q) / 1000);
        };
    };

    // X.T: xt[k][i] = x[i][k]
    let mut xt: Array<u128> = ArrayTrait::new();  // row-major [20][10].
    for k in 0_u32..20 {
        for i in 0_u32..10 {
            xt.append(*x.at(i * 20 + k));
        };
    };

    let mut result: Array<u128> = ArrayTrait::new();
    for k in 0_u32..20 {
        for i in 0_u32..10 {
            for j in 0_u32..10 {
                let a: u128 = *xt.at(k * 10 + i);
                let b: u128 = *xt.at(k * 10 + j);
                result.append((a * b) / Q);
            };
        };
    };

    // Claimed: compute the same expression directly. Tolerance ±2 in Q
    // absorbs the two truncating divisions (the 1/1000 scaling and qmul rescale).
    let tol: u128 = 2;
    for k in 0_u32..20 {
        for i in 0_u32..10 {
            for j in 0_u32..10 {
                let xa_num: u128 = ((i + 1) * 10 + (k + 1)).into();
                let xb_num: u128 = ((j + 1) * 10 + (k + 1)).into();
                let xa: u128 = (xa_num * Q) / 1000;
                let xb: u128 = (xb_num * Q) / 1000;
                let expected: u128 = (xa * xb) / Q;
                let got: u128 = *result.at((k * 10 + i) * 10 + j);
                let diff: u128 = if got >= expected { got - expected } else { expected - got };
                assert!(diff <= tol);
            };
        };
    };
}
