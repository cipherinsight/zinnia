// npbench_polybench_gramschmidt_gramschmidt__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each dot rescales once; updates materialize full column products. Constant lower-triangle outputs remain omitted.
// @outputs: float[NR * NC]; repeat[k < NC](const float[k] = 0 | float[NC - k])

use noir_hw::{check_inputs, divmod_pos, mul, rescale, sqrt, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global NR: u32 = 14; // = A_0
global NC: u32 = 12; // = A_1
global UPPER: u32 = NC * (NC + 1) / 2;

fn main(A: [[Field; NC]; NR]) -> pub ([[Field; NC]; NR], [Field; UPPER]) {
    for i in 0..NR { check_inputs(A[i]); }
    let mut a = A;
    let mut q = [[0; NC]; NR];
    let mut r = [[0; NC]; NC];
    for k in 0..NC {
        let mut nrm = 0;
        for i in 0..NR { nrm += a[i][k] * a[i][k]; }
        r[k][k] = sqrt(rescale(nrm));
        for i in 0..NR { q[i][k] = noir_hw::div(a[i][k], r[k][k]); }
        for j in k+1..NC {
            let mut acc = 0;
            for i in 0..NR { acc += q[i][k] * a[i][j]; }
            r[k][j] = rescale(acc);
            let mut products = [0; NR];
            for i in 0..NR { products[i] = mul(q[i][k], r[k][j]); }
            for i in 0..NR { a[i][j] -= products[i]; }
        }
    }
    let mut upper = [0; UPPER];
    let mut pos = 0;
    for k in 0..NC { for j in k..NC { upper[pos] = r[k][j]; pos += 1; } }
    (q, upper)
}
