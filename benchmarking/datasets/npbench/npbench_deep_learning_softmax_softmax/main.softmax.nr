// npbench_deep_learning_softmax_softmax__softmax (NPBench deep_learning/softmax), Noir baseline.
// Kernel: softmax over the last axis of x: Float[N, H, SN, SM]:
// tmp_max = np.max(x, axis=-1, keepdims=True), tmp_out = np.exp(x - tmp_max),
// tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True), tmp_out / tmp_sum.
// Each row's maximum is `max_of` (a hinted maximum, every gap max - x checked
// to [0, 2^64); 89,623 gates against 101,015 for a chain of comparisons),
// and those checks are the precondition of `exp_neg` (the shared gadget,
// NPB_BASELINES.md section 5, rule 5; its own check of the gap is the same
// constraint), so exp(x - max) is exp_neg(max - x). The sum is exact; each
// quotient is a float division by the sum, positive (the row's maximum
// contributes exp(0)), so `divmod_pos` needs no sign test.
// @outputs: float[N * H * SN * SM]

use noir_hw::{check_inputs, divmod_pos, exp_neg, max_of, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 2; // = x_0
global H: u32 = 2; // = x_1
global SN: u32 = 16; // = x_2
global SM: u32 = 16; // = x_3

fn softmax_row(row: [Field; SM]) -> [Field; SM] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(row);
    let largest = max_of(row);
    let mut exps: [Field; SM] = [0; SM];
    let mut total: Field = 0;
    for k in 0..SM {
        exps[k] = exp_neg(largest - row[k]);
        total += exps[k];
    }
    let mut out: [Field; SM] = [0; SM];
    for k in 0..SM {
        let (quotient, _) = divmod_pos(exps[k] * SCALE, total);
        out[k] = quotient;
    }
    out
}

fn main(x: [[[[Field; SM]; SN]; H]; N]) -> pub [[[[Field; SM]; SN]; H]; N] {
    let mut out = x;
    for a in 0..N {
        for b in 0..H {
            for c in 0..SN {
                out[a][b][c] = softmax_row(x[a][b][c]);
            }
        }
    }
    out
}
