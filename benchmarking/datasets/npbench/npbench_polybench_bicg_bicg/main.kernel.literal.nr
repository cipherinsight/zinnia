// npbench_polybench_bicg_bicg__kernel, Noir baseline: literal translation of
// kernel.py (LITERAL_BASELINES.md).
// Kernel: return (r @ A, A @ p) on A: Float[M, N], p: Float[N], r: Float[M].
// Each `@` is one NumPy reduction call, so L5's dot-product row applies: the
// defining sum over the shared axis, with the single allowed rescale, one
// call per output element, in the tuple's source order.
// @outputs: float[N]; float[M]

use noir_hw::{check_inputs, dot, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 16; // = A_0
global N: u32 = 13; // = A_1

fn main(A: [[Field; N]; M], p: [Field; N], r: [Field; M]) -> pub ([Field; N], [Field; M]) {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(A[i]);
    }
    check_inputs(p);
    check_inputs(r);

    // r @ A: element j is the dot of r with A's column j.
    let mut s: [Field; N] = [0; N];
    for j in 0..N {
        let mut acc: Field = 0;
        for i in 0..M {
            acc += r[i] * A[i][j];
        }
        s[j] = rescale(acc);
    }
    // A @ p: element i is the dot of A's row i with p.
    let mut t: [Field; M] = [0; M];
    for i in 0..M {
        t[i] = dot(A[i], p);
    }
    (s, t)
}
