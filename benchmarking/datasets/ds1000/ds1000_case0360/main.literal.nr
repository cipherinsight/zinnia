// DS-1000 case 0360, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel: `a = np.delete(a, 2, axis=0)` on Float[4, 4]. The deleted row is a
// Python-level constant (L3: compile-time numbers), so which cells are kept
// is resolved at compile time; the circuit does no work beyond exposing
// the kept input cells.
// @outputs: float[12]

fn main(a: [Field; 16]) -> pub [Field; 12] {
    // a = np.delete(a, 2, axis=0)
    let del_row: u32 = 2;
    let mut out: [Field; 12] = [0; 12];
    let mut k: u32 = 0;
    for i in 0..4 {
        if i != del_row {
            for j in 0..4 {
                out[k] = noir_hw::checked(a[i * 4 + j]);
                k += 1;
            }
        }
    }
    out
}
