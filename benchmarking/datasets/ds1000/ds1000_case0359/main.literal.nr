// DS-1000 case 0359, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel: `a = np.delete(a, 2, axis=1)` on Float[3, 3]. The deleted column is
// a Python-level constant (L3: compile-time numbers), so which cells are
// kept is resolved at compile time; the circuit does no work beyond
// exposing the kept input cells.
// @outputs: float[6]

fn main(a: [Field; 9]) -> pub [Field; 6] {
    // a = np.delete(a, 2, axis=1)
    let del_col: u32 = 2;
    let mut out: [Field; 6] = [0; 6];
    let mut k: u32 = 0;
    for i in 0..3 {
        for j in 0..3 {
            if j != del_col {
                out[k] = noir_hw::checked(a[i * 3 + j]);
                k += 1;
            }
        }
    }
    out
}
