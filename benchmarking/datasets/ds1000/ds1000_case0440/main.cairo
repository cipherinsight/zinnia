// DS-1000 case 0440 — diag-sqrt-transpose stacking.
// Y[i][j][j] = ((i+1)*(j+1))² for j==k else 0. N=2, M=K=3.
// Expected X (unquantised) = [[1, 2], [2, 4], [3, 6]].
// Quantisation ×2^32 to match Zinnia. q_sqrt(q_x) = sqrt(q_x * Q) (mirrors
// FixedPointChip's fixed-point sqrt: result² ≈ q_x · Q, then result / Q² · Q
// rescales back into Q-space).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;
    let n: u32 = 2;
    let m: u32 = 3;
    let k: u32 = 3;
    let mut y: Array<u128> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < n {
        let mut j: u32 = 0;
        while j < m {
            let mut kk: u32 = 0;
            while kk < k {
                if j == kk {
                    let v: u128 = ((i + 1) * (j + 1)).into();
                    // v² stored in Q-scale: (v*v) * Q.
                    y.append(v * v * Q);
                } else {
                    y.append(0);
                };
                kk += 1;
            };
            j += 1;
        };
        i += 1;
    };
    let mut x: Array<u128> = ArrayTrait::new();
    let mut jj: u32 = 0;
    while jj < m {
        let mut ii: u32 = 0;
        while ii < n {
            let idx: u32 = ii * (m * k) + jj * k + jj;
            let q_x: u128 = *y.at(idx.try_into().unwrap());
            // Fixed-point sqrt: q_y = sqrt(q_x * Q) so (q_y/Q)² ≈ q_x/Q.
            let target: u128 = q_x * Q;
            let mut g: u128 = if target > 0 { target } else { 1 };
            let mut prev: u128 = 0;
            while g != prev {
                prev = g;
                g = (g + target / g) / 2;
            };
            x.append(g);
            ii += 1;
        };
        jj += 1;
    };
    // Expected X (unquantised) row-major = [1, 2, 2, 4, 3, 6]; each cell × Q.
    let exp_unq: Array<u128> = array![1, 2, 2, 4, 3, 6];
    for kk in 0_usize..6 { assert!(*x.at(kk) == *exp_unq.at(kk) * Q); };
}
