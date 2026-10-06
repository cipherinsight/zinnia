// DS-1000 case 0346, Noir baseline.
// Kernel: `a[row-1, :] *= multiply_number; np.cumsum(a[row-1, :])` on
// a: Float[8, 6], row: Integer, multiply_number: Float.
// Only the selected row reaches the output: each of its 6 values is a
// noir_hw::pick down a column (NumPy wrap and bounds for row - 1; any other
// row has no proof), scaled (6 rescaled products) and summed (exact prefix
// sums). The pick's zero tests are the same for every column, so the compiler
// builds them once (index_pos with native reads costs 96 gates more).
// @outputs: float[6]

use noir_hw::{mul, pick};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, multiply_number: Field) -> pub [Field; 6] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(row);
    noir_hw::check_input(multiply_number);
    let mut cumsum = [0; COLS];
    let mut acc: Field = 0;
    for j in 0..COLS {
        let mut column = [0; ROWS];
        for i in 0..ROWS {
            column[i] = a[i * COLS + j];
        }
        acc += mul(pick(column, row - 1), multiply_number);
        cumsum[j] = acc;
    }
    cumsum
}
