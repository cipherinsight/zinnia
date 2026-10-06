// DS-1000 case 0345, Noir baseline.
// Kernel: `a[:, col-1] *= multiply_number; np.cumsum(a[:, col-1])` on
// a: Float[8, 6], col: Integer, multiply_number: Float.
// Only the selected column reaches the output: each row's value is a
// noir_hw::pick (NumPy wrap and bounds for col - 1; any other col has no
// proof), scaled (8 rescaled products) and summed (exact prefix sums). The
// pick's zero tests are the same for every row, so the compiler builds them
// once (the same gate count as a hand-shared one-hot; index_pos with native
// reads costs 105 gates more).
// @outputs: float[8]

use noir_hw::{mul, pick};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], col: Field, multiply_number: Field) -> pub [Field; 8] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(col);
    noir_hw::check_input(multiply_number);
    let mut cumsum = [0; ROWS];
    let mut acc: Field = 0;
    for i in 0..ROWS {
        let mut row = [0; COLS];
        for j in 0..COLS {
            row[j] = a[i * COLS + j];
        }
        acc += mul(pick(row, col - 1), multiply_number);
        cumsum[i] = acc;
    }
    cumsum
}
