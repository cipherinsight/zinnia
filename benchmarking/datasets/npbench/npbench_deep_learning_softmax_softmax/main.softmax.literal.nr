// npbench_deep_learning_softmax_softmax__softmax: literal translation of kernel.py (LITERAL_BASELINES.md).
// Pairwise maximum, full subtraction, canonical exp, running sums, ordinary divisions.
// @outputs: float[N * H * SN * SM]
global N: u32 = 2; // = x_0
global H: u32 = 2; // = x_1
global SN: u32 = 16; // = x_2
global SM: u32 = 16; // = x_3
global ROWS: u32 = N * H * SN;
fn main(x: [[Field; SM]; ROWS]) -> pub [[Field; SM]; ROWS] {
    for i in 0..ROWS {
        noir_hw::check_inputs(x[i]);
    }
    let mut tmp_max = [0; ROWS];
    for i in 0..ROWS {
        let mut best = x[i][0];
        for j in 1..SM {
            best = noir_hw::max(best, x[i][j]);
        }
        tmp_max[i] = best;
    }
    let mut gaps = [[0; SM]; ROWS];
    for i in 0..ROWS {
        for j in 0..SM {
            gaps[i][j] = x[i][j] - tmp_max[i];
        }
    }
    let mut tmp_out = [[0; SM]; ROWS];
    for i in 0..ROWS {
        for j in 0..SM {
            tmp_out[i][j] = noir_hw::canonical_exp(gaps[i][j]);
        }
    }
    let mut tmp_sum = [0; ROWS];
    for i in 0..ROWS {
        let mut sum: Field = 0;
        for j in 0..SM {
            sum += tmp_out[i][j];
        }
        tmp_sum[i] = sum;
    }
    let mut result = [[0; SM]; ROWS];
    for i in 0..ROWS {
        for j in 0..SM {
            result[i][j] = noir_hw::div(tmp_out[i][j], tmp_sum[i]);
        }
    }
    result
}
