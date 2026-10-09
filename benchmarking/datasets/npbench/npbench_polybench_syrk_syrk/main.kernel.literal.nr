// npbench_polybench_syrk_syrk__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// The scalar alpha*A[i,k] is evaluated once at its source location, then broadcast; products precede the slice update.
// @outputs: float[N * N]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 14; // = A_0
global M: u32 = 10; // = A_1

fn main(alpha: Field, beta: Field, C: [[Field; N]; N], A: [[Field; M]; N]) -> pub [[Field; N]; N] {
    check_input(alpha); check_input(beta);
    for i in 0..N { check_inputs(C[i]); check_inputs(A[i]); }
    let mut c = C;
    for i in 0..N {
        for j in 0..i+1 { c[i][j] = mul(c[i][j], beta); }
        for k in 0..M {
            let scale = mul(alpha, A[i][k]);
            let mut products = [0; N];
            for j in 0..i+1 { products[j] = mul(scale, A[j][k]); }
            for j in 0..i+1 { c[i][j] += products[j]; }
        }
    }
    c
}
