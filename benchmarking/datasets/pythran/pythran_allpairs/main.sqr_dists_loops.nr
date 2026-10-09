// pythran_allpairs__sqr_dists_loops (Pythran allpairs), Noir baseline.
// Kernel: D[i, j] = np.sum((X[i] - Y[j]) ** 2) for every row i of
// X: Float[M, N] and row j of Y: Float[P, N]; the output is D: M x P.
// Each sum of squares is one reduction, rescaled once (NPB_BASELINES.md 5.3).
// Its raw value is computed as |x|^2 + |y|^2 - 2 x . y, which is the same
// integer as sum((x - y)^2): the squared norms are computed once per row, so
// each pair costs one inner product.
// @outputs: float[M * P]

use noir_hw::{check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.sqr_dists_loops.json.
global M: u32 = 16; // = X_0
global N: u32 = 16; // = X_1
global P: u32 = 16; // = Y_0

fn raw_dot(a: [Field; N], b: [Field; N]) -> Field {
    let mut acc: Field = 0;
    for k in 0..N {
        acc += a[k] * b[k];
    }
    acc
}

fn main(X: [[Field; N]; M], Y: [[Field; N]; P]) -> pub [[Field; P]; M] {
    // Input domain: every input element is a Q32 code in the window.
    let mut x_norms: [Field; M] = [0; M];
    for i in 0..M {
        check_inputs(X[i]);
        x_norms[i] = raw_dot(X[i], X[i]);
    }
    let mut y_norms: [Field; P] = [0; P];
    for j in 0..P {
        check_inputs(Y[j]);
        y_norms[j] = raw_dot(Y[j], Y[j]);
    }
    let mut result: [[Field; P]; M] = [[0; P]; M];
    for i in 0..M {
        for j in 0..P {
            result[i][j] = rescale(x_norms[i] + y_norms[j] - 2 * raw_dot(X[i], Y[j]));
        }
    }
    result
}
