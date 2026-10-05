// DS-1000 case 0439, Noir baseline.
// Kernel: `X.T[:, :, None] * X.T[:, None]` on Float[10, 20]: the Float[20, 10, 10]
// array r[k, i, j] = X[i, k] * X[j, k], each a rescaled product.
// The product is symmetric in (i, j) (the same rounded value), so each of the
// 20 * 55 unordered pairs is computed once and returned at both positions.
// @outputs: float[2000]

use noir_hw::mul;

fn main(X: [Field; 200]) -> pub [Field; 2000] {
    let mut r = [0; 2000];
    for k in 0..20 {
        for i in 0..10 {
            for j in i..10 {
                let p = mul(X[i * 20 + k], X[j * 20 + k]);
                r[k * 100 + i * 10 + j] = p;
                r[k * 100 + j * 10 + i] = p;
            }
        }
    }
    r
}
