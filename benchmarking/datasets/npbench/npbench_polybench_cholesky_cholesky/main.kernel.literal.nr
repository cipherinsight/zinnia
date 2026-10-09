// npbench_polybench_cholesky_cholesky__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Dots rescale once; ordinary division at each source division.
// @outputs: float[N * N]

use noir_hw::{check_inputs, div, rescale, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 15; // = A_0

fn main(A: [[Field; N]; N]) -> pub [[Field; N]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(A[i]);
    }
    let mut a = A;

    // A[0, 0] = np.sqrt(A[0, 0])
    a[0][0] = sqrt(a[0][0]);

    for i in 1..N {
        for j in 0..i {
            // A[i, j] -= np.dot(A[i, :j], A[j, :j])
            let mut acc: Field = 0;
            for k in 0..j {
                acc += a[i][k] * a[j][k];
            }
            let dot = if j == 0 { 0 } else { rescale(acc) };
            a[i][j] -= dot;
            // A[i, j] /= A[j, j]
            a[i][j] = div(a[i][j], a[j][j]);
        }
        // A[i, i] -= np.dot(A[i, :i], A[i, :i])
        let mut radicand = a[i][i];
        if i > 0 {
            let mut acc: Field = 0;
            for k in 0..i {
                acc += a[i][k] * a[i][k];
            }
            radicand -= rescale(acc);
        }
        // A[i, i] = np.sqrt(A[i, i])
        a[i][i] = sqrt(radicand);
    }
    a
}
