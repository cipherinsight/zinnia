// DS-1000 case 0319, Noir baseline.
// Kernel: `c = np.empty_like(p); c[p] = np.arange(8); a[c, :, :]` on
// a: Float[8, 6, 8], permutation: Integer[8].
// Precondition (enforced): `permutation` is a permutation of 0..8 after
// NumPy's negative-index wrap. Otherwise NumPy raises (an index outside
// [-8, 8)) or leaves entries of `c` uninitialized (`np.empty`), and there is
// no proof.
// Output plane j is plane c[j] of `a`, the k with permutation[k] = j:
// hit_k = [(p_k - j)(p_k - j + 8) == 0] (p_k is j or j - 8); exactly one k may
// hit, which over all j proves the precondition, and c[j] is that k (a select
// of loop constants, so the native read a[c[j]] needs no range check).
// @outputs: float[384]

fn main(a: [[Field; 48]; 8], permutation: [Field; 8]) -> pub [Field; 384] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    for r in 0..8 {
        noir_hw::check_inputs(a[r]);
    }
    noir_hw::check_inputs(permutation);
    let mut out: [Field; 384] = [0; 384];
    for j in 0..8 {
        let mut total: Field = 0;
        let mut c: u32 = 0;
        for k in 0..8 {
            let d = permutation[k] - j as Field;
            let hit = d * (d + 8) == 0;
            total += hit as Field;
            if hit {
                c = k;
            }
        }
        assert(total == 1, "permutation is not a permutation of 0..8");
        let plane = a[c];
        for m in 0..48 {
            out[j * 48 + m] = plane[m];
        }
    }
    out
}
