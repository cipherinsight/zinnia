// npbench_polybench_symm_symm__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Keep temp2 and the row expression after the j loop; recompute alpha*B for each source occurrence.
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
    let mut temp2 = [0; N];
    let mut c = C;
    for i in 0..M { for j in 0..N { c[i][j] = mul(c[i][j],beta); } }
    for i in 0..M {
        for j in 0..N {
            let ab = mul(alpha,B[i][j]);
            let mut terms = [0; M];
            for k in 0..i { terms[k] = mul(ab,A[i][k]); }
            for k in 0..i { c[k][j] += terms[k]; }
            let mut raw: Field = 0;
            for k in 0..i { raw += B[k][j]*A[i][k]; }
            temp2[j] = if i == 0 { 0 } else { rescale(raw) };
        }
        let mut ab = [0; N];
        for j in 0..N { ab[j] = mul(alpha,B[i][j]); }
        let mut diagonal = [0; N];
        for j in 0..N { diagonal[j] = mul(ab[j],A[i][i]); }
        let mut at = [0; N];
        for j in 0..N { at[j] = mul(alpha,temp2[j]); }
        let mut update = [0; N];
        for j in 0..N { update[j] = diagonal[j]+at[j]; }
        for j in 0..N { c[i][j] += update[j]; }
    }
    c
}
