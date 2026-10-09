// pythran_cronbach__cronbach: literal translation of kernel.py (LITERAL_BASELINES.md).
// Variance uses mean, full deviations, rounded squares, sum, then ddof=1 division.

// @outputs: float[1]

use noir_hw::{check_inputs, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global ROWS: u32 = 16; // = itemscores_0
global COLS: u32 = 16; // = itemscores_1
global DDOF_DIVISOR: u32 = COLS - 1;
// round(ROWS / (ROWS - 1) * 2^32), rounding half up.
global RATIO_Q: Field =
    ((2 * 4294967296 * (ROWS as u64) + (ROWS as u64 - 1)) / (2 * (ROWS as u64 - 1))) as Field;

fn main(itemscores: [[Field; COLS]; ROWS]) -> pub [Field; 1] {
    for i in 0..ROWS {
        check_inputs(itemscores[i]);
    }
    let mut means = [0; ROWS];
    for i in 0..ROWS {
        let mut total: Field = 0;
        for j in 0..COLS {
            total += itemscores[i][j];
        }
        means[i] = noir_hw::divmod_pos(total, COLS as Field).0;
    }
    let mut deviations = [[0; COLS]; ROWS];
    for i in 0..ROWS {
        for j in 0..COLS {
            deviations[i][j] = itemscores[i][j] - means[i];
        }
    }
    let mut squares = [[0; COLS]; ROWS];
    for i in 0..ROWS {
        for j in 0..COLS {
            squares[i][j] = mul(deviations[i][j], deviations[i][j]);
        }
    }
    let mut itemvars = [0; ROWS];
    for i in 0..ROWS {
        let mut total: Field = 0;
        for j in 0..COLS {
            total += squares[i][j];
        }
        itemvars[i] = noir_hw::divmod_pos(total, DDOF_DIVISOR as Field).0;
    }
    let mut tscores = [0; COLS];
    for j in 0..COLS {
        for i in 0..ROWS {
            tscores[j] += itemscores[i][j];
        }
    }
    let mut itemvars_sum: Field = 0;
    for i in 0..ROWS {
        itemvars_sum += itemvars[i];
    }
    let mut total: Field = 0;
    for j in 0..COLS {
        total += tscores[j];
    }
    let mean = noir_hw::divmod_pos(total, COLS as Field).0;
    let mut deviations = [0; COLS];
    for j in 0..COLS {
        deviations[j] = tscores[j] - mean;
    }
    let mut squares = [0; COLS];
    for j in 0..COLS {
        squares[j] = mul(deviations[j], deviations[j]);
    }
    let mut total: Field = 0;
    for j in 0..COLS {
        total += squares[j];
    }
    let tvar = noir_hw::divmod_pos(total, DDOF_DIVISOR as Field).0;
    let ratio = noir_hw::canonical_div_nonzero(itemvars_sum, tvar);
    [mul(RATIO_Q, SCALE - ratio)]
}
