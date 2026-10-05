// DS-1000 case 501 — Cairo baseline.
// kernel.py:
//     result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
//   a: Float[13,13,2], b: Integer[13,13]; output: Float[13,13].
//
// Quantisation ×2^32 (Q) to match Zinnia's `precision_bits = 32`. Each f64
// literal `v` is encoded as `(v * Q) as u128` via `(int * Q) / denom` to
// avoid pre-computing each entry by hand. Equality is exact in Q-scale
// because no qmul is needed (this kernel is a pure gather).
//
// Sample inputs (matches the SP1 / Risc0 drivers):
//   a[i][j][k] = (i*26 + j*2 + k + 1) / 100
//   b[i][j]    = (i + j) % 2
//   expected[i][j] = a[i][j][b[i][j]]

#[executable]
pub fn main() {
    const Q: u128 = 4_294_967_296;

    // Build flattened a[13*13*2] in Q-scale.
    let mut a: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            for k in 0_u128..2 {
                let num: u128 = i * 26 + j * 2 + k + 1;
                a.append((num * Q) / 100);
            };
        };
    };

    // Build flattened b[13*13] (raw integer indices).
    let mut b: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            b.append((i + j) % 2);
        };
    };

    // Build flattened expected[13*13] in Q-scale.
    let mut expected: Array<u128> = ArrayTrait::new();
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = (i + j) % 2;
            let num: u128 = i * 26 + j * 2 + bij + 1;
            expected.append((num * Q) / 100);
        };
    };

    // Gather + assert.
    for i in 0_u128..13 {
        for j in 0_u128..13 {
            let bij: u128 = *b.at((i * 13 + j).try_into().unwrap());
            let a_idx: u32 = (i * 26 + j * 2 + bij).try_into().unwrap();
            let v: u128 = *a.at(a_idx);
            let e: u128 = *expected.at((i * 13 + j).try_into().unwrap());
            assert!(v == e);
        };
    };
}
