// npbench_polybench_gramschmidt_gramschmidt__kernel (NPBench polybench/gramschmidt), Noir baseline.
// Kernel: modified Gram-Schmidt on A: Float[NR, NC]; the output is the tuple
// (Q, R), Q: [NR, NC], R: [NC, NC]. For each column k:
// R[k, k] = sqrt(np.dot(A[:, k], A[:, k])), Q[:, k] = A[:, k] / R[k, k], and
// for each later column j: R[k, j] = np.dot(Q[:, k], A[:, j]),
// A[:, j] -= Q[:, k] * R[k, j].
// Each dot product is one reduction, rescaled once; the root is the exact
// floor square root; the quotients divide by the root, positive (a zero column
// has no proof: NumPy gives nan); the products of the update are rounded one
// by one. R starts as zeros and only its upper triangle is written: the lower
// triangle is a constant output, not public (CONVENTIONS 1.3); the public part
// of R is its upper triangle, row by row.
// @outputs: float[NR * NC]; repeat[k < NC](const float[k] = 0 | float[NC - k])

use noir_hw::{check_inputs, divmod_pos, mul, rescale, sqrt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global NR: u32 = 14; // = A_0
global NC: u32 = 12; // = A_1
global UPPER: u32 = NC * (NC + 1) / 2;

fn main(A: [[Field; NC]; NR]) -> pub ([[Field; NC]; NR], [Field; UPPER]) {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..NR {
        check_inputs(A[i]);
    }
    let mut a = A;
    let mut q: [[Field; NC]; NR] = [[0; NC]; NR];
    let mut r: [Field; UPPER] = [0; UPPER];
    let mut pos: u32 = 0; // position of R[k, k] in the upper triangle
    for k in 0..NC {
        let mut acc: Field = 0;
        for i in 0..NR {
            acc += a[i][k] * a[i][k];
        }
        let r_kk = sqrt(rescale(acc));
        r[pos] = r_kk;
        for i in 0..NR {
            // A[i, k] / R[k, k] with R[k, k] > 0: floor(A[i, k] 2^32 / R[k, k]).
            let (quotient, _) = divmod_pos(a[i][k] * SCALE, r_kk);
            q[i][k] = quotient;
        }
        for j in k + 1..NC {
            let mut dot: Field = 0;
            for i in 0..NR {
                dot += q[i][k] * a[i][j];
            }
            let r_kj = rescale(dot);
            r[pos + j - k] = r_kj;
            for i in 0..NR {
                a[i][j] -= mul(q[i][k], r_kj);
            }
        }
        pos += NC - k;
    }
    (q, r)
}
