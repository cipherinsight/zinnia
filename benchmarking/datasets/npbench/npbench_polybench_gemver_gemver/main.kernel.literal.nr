// npbench_polybench_gemver_gemver__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize both outer products and every intermediate array in expression order; dot uses one rescale.
// @outputs: float[N * M]; float[N]; float[M]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = A_0
global M: u32 = 16; // = A_1

fn main(
    alpha: Field,
    beta: Field,
    A: [[Field; M]; N],
    u1: [Field; N],
    v1: [Field; M],
    u2: [Field; N],
    v2: [Field; M],
    w: [Field; N],
    x: [Field; M],
    y: [Field; N],
    z: [Field; M],
) -> pub ([[Field; M]; N], [Field; N], [Field; M]) {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..N {
        check_inputs(A[i]);
    }
    check_inputs(u1);
    check_inputs(v1);
    check_inputs(u2);
    check_inputs(v2);
    check_inputs(w);
    check_inputs(x);
    check_inputs(y);
    check_inputs(z);
    let mut first = [[0; M]; N];
    for i in 0..N { for j in 0..M { first[i][j] = mul(u1[i],v1[j]); } }
    let mut second = [[0; M]; N];
    for i in 0..N { for j in 0..M { second[i][j] = mul(u2[i],v2[j]); } }
    let mut updates = [[0; M]; N];
    for i in 0..N { for j in 0..M { updates[i][j] = first[i][j]+second[i][j]; } }
    let mut a = A;
    for i in 0..N { for j in 0..M { a[i][j] += updates[i][j]; } }
    let mut by = [0; N];
    for i in 0..N { by[i] = mul(beta,y[i]); }
    let mut product = [0; M];
    for j in 0..M { let mut raw: Field = 0; for i in 0..N { raw += by[i]*a[i][j]; } product[j] = rescale(raw); }
    let mut update = [0; M];
    for j in 0..M { update[j] = product[j]+z[j]; }
    let mut xx = x;
    for j in 0..M { xx[j] += update[j]; }
    let mut aa = [[0; M]; N];
    for i in 0..N { for j in 0..M { aa[i][j] = mul(alpha,a[i][j]); } }
    let mut product2 = [0; N];
    for i in 0..N { let mut raw: Field = 0; for j in 0..M { raw += aa[i][j]*xx[j]; } product2[i] = rescale(raw); }
    let mut ww = w;
    for i in 0..N { ww[i] += product2[i]; }
    (a,ww,xx)
}
