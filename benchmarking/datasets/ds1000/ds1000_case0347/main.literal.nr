// DS-1000 case 0347, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel:
//   a[row-1, :] /= divide_number
//   result = np.multiply.reduce(a[row-1, :])
// on a: Float[8, 6], row: Integer, divide_number: Float. Row write and read
// as in case 0346 (row - 1 in [-8, 8), computed and checked once per
// statement, no CSE across statements; the write uses the local `row_hits`
// helper, the read uses noir_hw::index_pos once per column). The division is
// noir_hw::div, called plainly at each of the 48 positions (unlike the
// optimized baseline, this does not hoist the divisor's sign test: any
// sharing of that check across positions is the compiler's own SSA pass, not
// something this translation does by hand). np.multiply.reduce is a running
// product left to right (L5), one rescaled product per step, seeded with
// the first element (no identity-element multiplication, matching NumPy's
// reduce).
// @outputs: float[1]

use noir_hw::{div, index_pos, mul};

global ROWS: u32 = 8;
global COLS: u32 = 6;

// The hit vector of a NumPy row index idx over len rows, for a
// private-index write (no shared gadget writes by a private index): row i
// is hit when idx is i or i - len (the wrapped negative index). Asserting
// exactly one hit pins idx to a valid NumPy index (any other value has no
// witness, as NumPy raises IndexError). Shared across the COLS columns the
// write touches, since the source statement applies the one row index to
// every column.
fn row_hits(idx: Field, len: u32) -> [bool; ROWS] {
    let mut hits: [bool; ROWS] = [false; ROWS];
    let mut total: Field = 0;
    for i in 0..len {
        let hit = (idx == i as Field) | (idx == i as Field - len as Field);
        hits[i] = hit;
        total += hit as Field;
    }
    assert(total == 1, "index out of bounds");
    hits
}

fn main(a: [Field; 48], row: Field, divide_number: Field) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(row);
    noir_hw::check_input(divide_number);

    // a[row-1, :] /= divide_number
    let idx1 = row - 1;
    let hits1 = row_hits(idx1, ROWS);
    let mut a = a;
    for i in 0..ROWS {
        for j in 0..COLS {
            let old = a[i * COLS + j];
            let quotient = div(old, divide_number);
            a[i * COLS + j] = if hits1[i] { quotient } else { old };
        }
    }

    // result = np.multiply.reduce(a[row-1, :])
    let idx2 = row - 1; // recomputed: no CSE across statements
    let mut row_values: [Field; 6] = [0; COLS];
    for j in 0..COLS {
        let mut column: [Field; 8] = [0; ROWS];
        for i in 0..ROWS {
            column[i] = a[i * COLS + j];
        }
        row_values[j] = column[index_pos::<8>(idx2, ROWS)];
    }
    let mut product = row_values[0];
    for j in 1..COLS {
        product = mul(product, row_values[j]);
    }
    product
}
