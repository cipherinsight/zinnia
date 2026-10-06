// DS-1000 case 0347, Noir baseline.
// Kernel: `a[row-1, :] /= divide_number; np.multiply.reduce(a[row-1, :])` on
// a: Float[8, 6], row: Integer, divide_number: Float.
// Only the selected row reaches the output: each of its 6 values is a
// noir_hw::pick down a column (NumPy wrap and bounds for row - 1; the zero
// tests are shared by the compiler), divided by divide_number (6 float
// divisions sharing one sign test of the divisor; a zero divisor has no
// proof) and multiplied out left to right (5 rescaled products).
// @outputs: float[1]

use noir_hw::{divmod_pos, is_neg, mul, pick, SCALE};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, divide_number: Field) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(row);
    noir_hw::check_input(divide_number);
    // noir_hw::div with the divisor's sign test hoisted: the quotient
    // floor(x * 2^32 / d) with remainder in [0, |d|) is the floor quotient by
    // |d|, negated for d < 0.
    let negative = is_neg(divide_number);
    let magnitude = if negative { -divide_number } else { divide_number };
    let mut quotients = [0; COLS];
    for j in 0..COLS {
        let mut column = [0; ROWS];
        for i in 0..ROWS {
            column[i] = a[i * COLS + j];
        }
        let (q, _r) = divmod_pos(pick(column, row - 1) * SCALE, magnitude);
        quotients[j] = if negative { -q } else { q };
    }
    let mut product = quotients[0];
    for j in 1..COLS {
        product = mul(product, quotients[j]);
    }
    product
}
