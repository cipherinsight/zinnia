// npbench_deep_learning_mlp_mlp__mlp (NPBench deep_learning/mlp), Noir baseline.
// Strict comparison: transcendental calls use the frozen compiler arithmetic,
// including its coefficients and each Q32 rounding point (canonical APIs).
// Kernel: a three-layer perceptron on input: Float[N, C_IN]:
// x = relu(input @ w1 + b1), x = relu(x @ w2 + b2), softmax(x @ w3 + b3) over
// the last axis, with relu(x) = np.maximum(x, 0) and softmax as in the
// softmax entry (exp(x - max) / sum).
// Each dot is rescaled and window-checked before its exact bias addition.
// ReLU and the final maximum use the frozen Noir gap-comparison domain;
// they do not impose an extra signed64 bound on the biased value.
// Softmax: each row's maximum is `max_of`, whose gap checks are the
// precondition of `exp_neg` (the shared gadget, NPB_BASELINES.md section 5,
// rule 5); the sum is exact and each quotient divides by it, positive (the
// row's maximum contributes exp(0)), so `divmod_pos` needs no sign test.
// @outputs: float[N * S2]

use noir_hw::{check_inputs, divmod_pos, canonical_exp_neg as exp_neg, canonical_gt, rescale, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = input_0
global C_IN: u32 = 2; // = input_1
global S0: u32 = 16; // = w1_1
global S1: u32 = 2; // = w2_1
global S2: u32 = 2; // = w3_1

// x @ w + b for x: [R, K], w: [K, C], b: [C], each element rescaled once.
fn dense<let R: u32, let K: u32, let C: u32>(x: [[Field; K]; R], w: [[Field; C]; K], b: [Field; C]) -> [[Field; C]; R] {
    let mut out: [[Field; C]; R] = [[0; C]; R];
    for r in 0..R {
        for c in 0..C {
            let mut acc: Field = 0;
            for k in 0..K {
                acc += x[r][k] * w[k][c];
            }
            out[r][c] = rescale(acc) + b[c];
        }
    }
    out
}

// np.maximum(x, 0) elementwise.
fn relu<let R: u32, let C: u32>(x: [[Field; C]; R]) -> [[Field; C]; R] {
    let mut out = x;
    for r in 0..R {
        for c in 0..C {
            if !canonical_gt(x[r][c], 0) {
                out[r][c] = 0;
            }
        }
    }
    out
}

fn main(
    input: [[Field; C_IN]; N],
    w1: [[Field; S0]; C_IN],
    b1: [Field; S0],
    w2: [[Field; S1]; S0],
    b2: [Field; S1],
    w3: [[Field; S2]; S1],
    b3: [Field; S2],
) -> pub [[Field; S2]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for r in 0..N {
        check_inputs(input[r]);
    }
    for k in 0..C_IN {
        check_inputs(w1[k]);
    }
    check_inputs(b1);
    for k in 0..S0 {
        check_inputs(w2[k]);
    }
    check_inputs(b2);
    for k in 0..S1 {
        check_inputs(w3[k]);
    }
    check_inputs(b3);

    let x = relu(dense(input, w1, b1));
    let x = relu(dense(x, w2, b2));
    let x = dense(x, w3, b3);
    let mut out: [[Field; S2]; N] = [[0; S2]; N];
    for r in 0..N {
        let mut largest = x[r][0];
        for c in 1..S2 {
            if canonical_gt(x[r][c], largest) { largest = x[r][c]; }
        }
        let mut exps: [Field; S2] = [0; S2];
        let mut total: Field = 0;
        for c in 0..S2 {
            exps[c] = exp_neg(largest - x[r][c]);
            total += exps[c];
        }
        for c in 0..S2 {
            let (quotient, _) = divmod_pos(exps[c] * SCALE, total);
            out[r][c] = quotient;
        }
    }
    out
}
