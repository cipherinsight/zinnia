// npbench_polybench_symm_symm__kernel (NPBench polybench/symm), Noir baseline.
// Kernel: on alpha, beta: Float, C: Float[M, N], A: Float[M, M],
// B: Float[M, N], in place (the output is C after the call):
//     C *= beta
//     for i: for j:
//         C[:i, j] += alpha * B[i, j] * A[i, :i]
//         temp2[j] = B[:i, j] @ A[i, :i]
//     C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2   (inside the loop over i)
// In NumPy's evaluation order, every product is rounded: alpha * B[i, j] (the
// same product as row i of alpha * B[i, :], computed once), then its products
// with A[i, :i] and A[i, i], and alpha * temp2[j]; temp2[j] is one reduction,
// rescaled once; for i = 0 it is the empty 0.0, and alpha * 0.0 is 0. The
// additions are exact. Only the lower triangle of A (with the diagonal) is
// read, and only it is checked.
// @outputs: float[M * N]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 10; // = C_0
global N: u32 = 13; // = C_1

fn main(alpha: Field, beta: Field, C: [[Field; N]; M], A: [[Field; M]; M], B: [[Field; N]; M]) -> pub [[Field; N]; M] {
    // Input domain: every element read is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..M {
        check_inputs(C[i]);
        check_inputs(B[i]);
        for k in 0..i + 1 {
            check_input(A[i][k]);
        }
    }
    let mut c: [[Field; N]; M] = [[0; N]; M];
    for i in 0..M {
        for j in 0..N {
            c[i][j] = mul(C[i][j], beta);
        }
    }
    for i in 0..M {
        for j in 0..N {
            let alpha_b = mul(alpha, B[i][j]);
            for k in 0..i {
                c[k][j] += mul(alpha_b, A[i][k]);
            }
            let mut update = mul(alpha_b, A[i][i]);
            if i > 0 {
                let mut acc: Field = 0;
                for k in 0..i {
                    acc += B[k][j] * A[i][k];
                }
                update += mul(alpha, rescale(acc));
            }
            c[i][j] += update;
        }
    }
    c
}
