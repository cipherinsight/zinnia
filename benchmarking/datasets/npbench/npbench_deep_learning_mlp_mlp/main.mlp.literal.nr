// npbench_deep_learning_mlp_mlp__mlp: literal translation of kernel.py (LITERAL_BASELINES.md).
// Matmuls rescale before separate bias arrays. ReLU/softmax are source helpers; max and sum scan in index order, canonical exp per element.
// @outputs: float[N * S2]

use noir_hw::{check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = input_0
global C_IN: u32 = 2; // = input_1
global S0: u32 = 16; // = w1_1
global S1: u32 = 2; // = w2_1
global S2: u32 = 2; // = w3_1

// L5 matrix product: complete output, then bias is computed separately.

fn matmul<let R: u32, let K: u32, let C: u32>(
    x: [[Field; K]; R],
    w: [[Field; C]; K],
) -> [[Field; C]; R] {
    let mut result = [[0; C]; R];
    for r in 0..R {
        for c in 0..C {
            let mut acc = 0;
            for k in 0..K {
                acc += x[r][k] * w[k][c];
            }
            result[r][c] = rescale(acc);
        }
    }
    result
}
fn bias<let R: u32, let C: u32>(x: [[Field; C]; R], b: [Field; C]) -> [[Field; C]; R] {
    let mut result = [[0; C]; R];
    for r in 0..R {
        for c in 0..C {
            result[r][c] = x[r][c] + b[c];
        }
    }
    result
}
fn relu<let R: u32, let C: u32>(x: [[Field; C]; R]) -> [[Field; C]; R] {
    let mut result = [[0; C]; R];
    for r in 0..R {
        for c in 0..C {
            result[r][c] = noir_hw::max(x[r][c], 0);
        }
    }
    result
}
fn softmax(x: [[Field; S2]; N]) -> [[Field; S2]; N] {
    let mut tmp_max = [0; N];
    for r in 0..N {
        let mut best = x[r][0];
        for c in 1..S2 {
            best = noir_hw::max(best, x[r][c]);
        }
        tmp_max[r] = best;
    }
    let mut difference = [[0; S2]; N];
    for r in 0..N {
        for c in 0..S2 {
            difference[r][c] = x[r][c] - tmp_max[r];
        }
    }
    let mut tmp_out = [[0; S2]; N];
    for r in 0..N {
        for c in 0..S2 {
            tmp_out[r][c] = noir_hw::canonical_exp(difference[r][c]);
        }
    }
    let mut tmp_sum = [0; N];
    for r in 0..N {
        let mut acc = 0;
        for c in 0..S2 {
            acc += tmp_out[r][c];
        }
        tmp_sum[r] = acc;
    }
    let mut result = [[0; S2]; N];
    for r in 0..N {
        for c in 0..S2 {
            result[r][c] = noir_hw::div(tmp_out[r][c], tmp_sum[r]);
        }
    }
    result
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
    let x = relu(bias(matmul(input, w1), b1));
    let x = relu(bias(matmul(x, w2), b2));
    let x = softmax(bias(matmul(x, w3), b3));
    x
}
