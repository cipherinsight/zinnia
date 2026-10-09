// npbench_polybench_gemver_gemver__kernel (NPBench polybench/gemver), Noir baseline.
// Kernel: on A: Float[N, M], u1, u2, w, y: Float[N] and v1, v2, x, z:
// Float[M], in place: A += outer(u1, v1) + outer(u2, v2),
// x += beta * y @ A + z, w += alpha * A @ x; the outputs are A, w, x after
// the call (`mutated_outputs`, parameter order).
// Rounding follows NumPy's evaluation order: each element of an outer product,
// of beta * y and of alpha * A is a rounded product (NumPy materializes them),
// each element of a matrix-vector product is one reduction rescaled once:
// (beta * y) @ A and (alpha * A) @ x.
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
    // A += np.outer(u1, v1) + np.outer(u2, v2)
    let mut a = A;
    for i in 0..N {
        for j in 0..M {
            a[i][j] += mul(u1[i], v1[j]) + mul(u2[i], v2[j]);
        }
    }
    // x += beta * y @ A + z, i.e. x + ((beta * y) @ A + z)
    let mut beta_y: [Field; N] = [0; N];
    for i in 0..N {
        beta_y[i] = mul(beta, y[i]);
    }
    let mut x_new = x;
    for j in 0..M {
        let mut acc: Field = 0;
        for i in 0..N {
            acc += beta_y[i] * a[i][j];
        }
        x_new[j] += rescale(acc) + z[j];
    }
    // w += alpha * A @ x, i.e. w + (alpha * A) @ x
    let mut w_new = w;
    for i in 0..N {
        let mut acc: Field = 0;
        for j in 0..M {
            acc += mul(alpha, a[i][j]) * x_new[j];
        }
        w_new[i] += rescale(acc);
    }
    (a, w_new, x_new)
}
