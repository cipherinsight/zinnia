// DS-1000 case 0318, Noir baseline.
// Kernel: `c = np.empty_like(p); c[p] = np.arange(6); a[:, c]` on
// a: Float[8, 6], permutation: Integer[6].
// Precondition (enforced): `permutation` is a permutation of 0..6 after
// NumPy's negative-index wrap. Otherwise NumPy raises (an index outside
// [-6, 6)) or leaves entries of `c` uninitialized (`np.empty`), and there is
// no proof.
// Output column j is column c[j] of `a`, the k with permutation[k] = j: the
// one-hot hit[j][k] = [(p_k - j)(p_k - j + 6) == 0] (p_k is j or j - 6; each
// test made once, shared by the 8 rows) selects it. Each k hits at most one j,
// so "every column j has exactly one hit" proves the precondition. The one-hot
// gather needs no range check and is cheaper here than native indexing.
// @outputs: float[48]

fn main(a: [Field; 48], permutation: [Field; 6]) -> pub [Field; 48] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(permutation);
    let mut hit: [[Field; 6]; 6] = [[0; 6]; 6];
    for j in 0..6 {
        let mut total: Field = 0;
        for k in 0..6 {
            let d = permutation[k] - j as Field;
            hit[j][k] = (d * (d + 6) == 0) as Field;
            total += hit[j][k];
        }
        assert(total == 1, "permutation is not a permutation of 0..6");
    }
    let mut out: [Field; 48] = [0; 48];
    for i in 0..8 {
        for j in 0..6 {
            let mut v: Field = 0;
            for k in 0..6 {
                v += hit[j][k] * a[i * 6 + k];
            }
            out[i * 6 + j] = v;
        }
    }
    out
}
