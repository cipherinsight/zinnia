// npbench_polybench_trmm_trmm__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Preserve in-place loop order, then scale the complete B; an empty dot is constant zero.
// @outputs: float[NI * NJ]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 13; // = A_0
global NJ: u32 = 16; // = B_1

fn main(alpha: Field, A: [[Field; NI]; NI], B: [[Field; NJ]; NI]) -> pub [[Field; NJ]; NI] {
    // Input domain: every element read is a Q32 code in the window.
    check_input(alpha);
    for i in 0..NI {
        for k in 0..i {
            check_input(A[i][k]);
        }
        check_inputs(B[i]);
    }
    let mut b = B;
    for i in 0..NI { for j in 0..NJ {
        let mut raw: Field = 0;
        for k in i+1..NI { raw += A[k][i]*b[k][j]; }
        let dot = if i+1 == NI { 0 } else { rescale(raw) };
        b[i][j] += dot;
    } }
    for i in 0..NI { for j in 0..NJ { b[i][j] = mul(b[i][j],alpha); } }
    b
}
