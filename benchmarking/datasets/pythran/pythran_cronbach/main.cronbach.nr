// pythran_cronbach__cronbach (Pythran cronbach), Noir baseline.
// Kernel: Cronbach's alpha of `itemscores: Float[ROWS, COLS]`:
// ROWS / (ROWS - 1) * (1 - itemvars.sum() / tscores.var(ddof=1)) with
// itemvars the rows' variances (ddof=1) and tscores the column sums.
// A variance is NumPy's `_var`: the mean sum / n (a quotient by the integer
// constant n), the sum of squared deviations (one reduction, rescaled once:
// NPB_BASELINES.md section 5.3), divided by n - 1. ROWS / (ROWS - 1) is a
// float64 constant of the kernel, so the last step is one rounded product
// with its Q32 code (RATIO_Q). The total variance is a sum of squares, so the
// division by it is `divmod_pos` (a zero variance has no witness, as NumPy's
// result is not finite).
// @outputs: float[1]

use noir_hw::{check_inputs, divmod_pos, mul, rescale, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global ROWS: u32 = 16; // = itemscores_0
global COLS: u32 = 16; // = itemscores_1
// round(ROWS / (ROWS - 1) * 2^32), rounding half up.
global RATIO_Q: Field =
    ((2 * 4294967296 * (ROWS as u64) + (ROWS as u64 - 1)) / (2 * (ROWS as u64 - 1))) as Field;

// xs.var(ddof=1): sum((x - mean)^2) / (n - 1) with mean = sum(x) / n.
fn var_ddof1<let N: u32>(xs: [Field; N]) -> Field {
    let mut total: Field = 0;
    for i in 0..N {
        total += xs[i];
    }
    let (mean, _) = divmod_pos(total, N as Field);
    let mut squares: Field = 0;
    for i in 0..N {
        let deviation = xs[i] - mean;
        squares += deviation * deviation;
    }
    let (var, _) = divmod_pos(rescale(squares), (N - 1) as Field);
    var
}

fn main(itemscores: [[Field; COLS]; ROWS]) -> pub [Field; 1] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..ROWS {
        check_inputs(itemscores[i]);
    }
    let mut itemvars_sum: Field = 0;
    for i in 0..ROWS {
        itemvars_sum += var_ddof1(itemscores[i]);
    }
    let mut tscores: [Field; COLS] = [0; COLS];
    for j in 0..COLS {
        for i in 0..ROWS {
            tscores[j] += itemscores[i][j];
        }
    }
    let tvar = var_ddof1(tscores);
    let (ratio, _) = divmod_pos(itemvars_sum * SCALE, tvar);
    [mul(RATIO_Q, SCALE - ratio)]
}
