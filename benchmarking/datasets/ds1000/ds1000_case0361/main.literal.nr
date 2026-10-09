// DS-1000 case 0361, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel:
//   temp = np.array([0, 2])
//   a = np.delete(a, temp, axis=1)
// on Float[6, 6]. temp is a Python-level array literal (L3: compile-time
// numbers), so the deleted columns are resolved at compile time; the
// circuit does no work beyond exposing the kept input cells.
// @outputs: float[24]

fn main(a: [Field; 36]) -> pub [Field; 24] {
    // temp = np.array([0, 2])
    let temp = [0, 2];
    // a = np.delete(a, temp, axis=1)
    let mut out: [Field; 24] = [0; 24];
    let mut k: u32 = 0;
    for i in 0..6 {
        for j in 0..6 {
            if (j != temp[0]) & (j != temp[1]) {
                out[k] = noir_hw::checked(a[i * 6 + j]);
                k += 1;
            }
        }
    }
    out
}
