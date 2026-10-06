// DS-1000 case 0362, Noir baseline.
// Kernel: `del_col = [12, 14, 1, 9]` (a constant), keep the entries
// `<= a.shape[1] = 6` and subtract 1, then `np.delete(a, del_col, axis=1)` on
// Float[8, 6]: only column 0 is deleted, the output is a[:, 1:].
// @outputs: float[40]

fn main(a: [Field; 48]) -> pub [Field; 40] {
    let mut out = [0; 40];
    for i in 0..8 {
        for j in 0..5 {
            out[i * 5 + j] = noir_hw::checked(a[i * 6 + 1 + j]);
        }
    }
    out
}
