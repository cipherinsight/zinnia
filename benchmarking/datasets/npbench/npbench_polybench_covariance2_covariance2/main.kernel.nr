// npbench_polybench_covariance2_covariance2__kernel (NPBench polybench/covariance2), Noir baseline.
// Kernel: `np.cov(np.transpose(data))` on data: Float[NO, NV] (NV variables,
// NO observations), with the precondition 0 <= M <= 1024 on M: Integer;
// float_n is never read and not checked.
// NumPy's cov: the mean of each variable (sum / NO, a float division by the
// integer count: floor(sum / NO) in Q32), the deviations (exact), the product
// of the deviation matrix with its transpose (each element one reduction,
// rescaled once), then direct division by NO - 1. Multiplication by a
// quantized reciprocal changes Q32 rounding and is not used. Symmetric
// entries reuse the identical ordered products as exact commutative CSE.
// M is checked against the source precondition 0 <= M <= 1024.
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
    let mut deviations: [[Field; NO]; NV] = [[0; NO]; NV];
    for v in 0..NV {
        let mut total: Field = 0;
        for o in 0..NO {
            total += data[o][v];
        }
        let (mean, _) = divmod_pos(total, NO as Field);
        for o in 0..NO {
            deviations[v][o] = data[o][v] - mean;
        }
    }
    let mut cov: [[Field; NV]; NV] = [[0; NV]; NV];
    for a in 0..NV {
        for b in a..NV {
            let mut acc: Field = 0;
            for o in 0..NO {
                acc += deviations[a][o] * deviations[b][o];
            }
            let (c, _) = divmod_pos(rescale(acc), (NO - 1) as Field);
            cov[a][b] = c;
            cov[b][a] = c;
        }
    }
    cov
}
