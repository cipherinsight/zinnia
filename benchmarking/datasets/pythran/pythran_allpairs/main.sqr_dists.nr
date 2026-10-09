// pythran_allpairs__sqr_dists (Pythran allpairs), Noir baseline.
// Kernel: np.array([[np.sum((x - y) ** 2) for x in X] for y in Y]) on
// X: Float[M, N], Y: Float[P, N]; the output is P x M, row j holding the
// squared distances from Y[j] to every row of X.
// Each sum of squares is one reduction, rescaled once (NPB_BASELINES.md 5.3).
// Its raw value is computed as |x|^2 + |y|^2 - 2 x . y, which is the same
// integer as sum((x - y)^2): the squared norms are computed once per row, so
// each pair costs one inner product.
// @outputs: float[P * M]

use noir_hw::{check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.sqr_dists.json.
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

fn main(X: [[Field; N]; M], Y: [[Field; N]; P]) -> pub [[Field; M]; P] {
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
    let mut result: [[Field; M]; P] = [[0; M]; P];
    for j in 0..P {
        for i in 0..M {
            result[j][i] = rescale(x_norms[i] + y_norms[j] - 2 * raw_dot(X[i], Y[j]));
        }
    }
    result
}
