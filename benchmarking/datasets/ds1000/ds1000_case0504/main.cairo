// DS-1000 case 504 — Cairo baseline.
// kernel.py:
//     arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//     result = np.sum(arr)
//
// Quantisation ×2^32 to match Zinnia's `precision_bits = 32`. The kernel is a
// pure gather followed by an additive reduction (no qmul), so Q-scaled sums
// are exact and exact `==` is appropriate.
//
// Sample inputs:
//   a[i][j][k] = (i*65 + j*5 + k + 1) / 100
//   b[i][j]    = (i + j) % 5
//
// Expected:
//   sum over (i, j) of a[i][j][b[i][j]] in Q-scale (computed at compile time
//   below by closed-form summation; we also recompute inside main and compare
//   to a witness for parity with other baselines).

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..5 {
                let num: u128 = i * 65 + j * 5 + k + 1;
                a.append((num * Q) / 100);
            };
        };
    };

    let mut b: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            b.append((i + j) % 5);
        };
    };

    // Compute claimed sum in Q-scale by iterating once (witness).
    let mut claimed_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = (i + j) % 5;
            let num: u128 = i * 65 + j * 5 + bij + 1;
            claimed_q = claimed_q + (num * Q) / 100;
        };
    };

    // Re-perform the gather + sum and compare.
    let mut arr: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = *b.at((i * 13 + j).try_into().unwrap());
            let a_idx: u32 = (i * 65 + j * 5 + bij).try_into().unwrap();
            arr.append(*a.at(a_idx));
        };
    };

    let mut result_q: u128 = 0;
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let idx: u32 = (i * 13 + j).try_into().unwrap();
            result_q = result_q + *arr.at(idx);
        };
    };

    assert!(result_q == claimed_q);
}
