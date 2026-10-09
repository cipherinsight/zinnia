// npbench_polybench_covariance2_covariance2__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// np.cov uses means, full centered matrix, full dot matrix and division by NO-1; no symmetry reuse.
// @outputs: float[NV * NV]

use noir_hw::{check_inputs, divmod_pos, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NO: u32 = 16; // = data_0
global NV: u32 = 14; // = data_1
// round(2^32 / (NO - 1)), the Q32 code of 1 / (NO - 1).

fn main(M: Field, float_n: Field, data: [[Field; NV]; NO]) -> pub [[Field; NV]; NV] {
    let _ = float_n;
    // Precondition 0 <= M <= 1024 (14-bit checks: one limb).
    M.assert_max_bit_size::<14>();
    (1024 - M).assert_max_bit_size::<14>();
    // Input domain: every element of data is a Q32 code in the window.
    for o in 0..NO {
        check_inputs(data[o]);
    }
    let mut transposed = [[0; NO]; NV];
    for v in 0..NV { for o in 0..NO { transposed[v][o] = data[o][v]; } }
    let mut means = [0; NV];
    for v in 0..NV {
        let mut total: Field = 0;
        for o in 0..NO { total += transposed[v][o]; }
        means[v] = divmod_pos(total,NO as Field).0;
    }
    let mut deviations = transposed;
    for v in 0..NV { for o in 0..NO { deviations[v][o] = transposed[v][o]-means[v]; } }
    let mut products = [[0; NV]; NV];
    for a in 0..NV { for b in 0..NV {
        let mut raw: Field = 0;
        for o in 0..NO { raw += deviations[a][o]*deviations[b][o]; }
        products[a][b] = rescale(raw);
    } }
    let mut cov = [[0; NV]; NV];
    for a in 0..NV { for b in 0..NV { cov[a][b] = divmod_pos(products[a][b],(NO-1) as Field).0; } }
    cov
}
