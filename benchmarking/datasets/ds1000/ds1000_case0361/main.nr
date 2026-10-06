// DS-1000 case 0361, Noir baseline.
// Kernel: `np.delete(a, [0, 2], axis=1)` on Float[6, 6]: the outputs are input values.
// @outputs: float[24]

fn main(a: [Field; 36]) -> pub [Field; 24] {
    let cols = [1, 3, 4, 5];
    let mut out = [0; 24];
    for i in 0..6 {
        for j in 0..4 {
            out[i * 4 + j] = noir_hw::checked(a[i * 6 + cols[j]]);
        }
    }
    out
}
