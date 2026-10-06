// DS-1000 case 0360, Noir baseline.
// Kernel: `np.delete(a, 2, axis=0)` on Float[4, 4]: the outputs are input values.
// @outputs: float[12]

fn main(a: [Field; 16]) -> pub [Field; 12] {
    let mut out = [0; 12];
    for k in 0..8 {
        out[k] = noir_hw::checked(a[k]);
    }
    for k in 0..4 {
        out[8 + k] = noir_hw::checked(a[12 + k]);
    }
    out
}
