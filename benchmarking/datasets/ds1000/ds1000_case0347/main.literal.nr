// DS-1000 case 0347, literal translation of kernel.py (LITERAL_BASELINES.md).
// Gather the augmented-assignment slice, compute each element, then scatter by equality selects.
// Negative indices wrap; each source indexed expression is evaluated separately.
// Reduction is a running accumulator in index order.
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
    let mut values = [0; COLS];
    for j in 0..COLS {
        let mut column = [0; ROWS];
        for i in 0..ROWS { column[i] = a[i*COLS+j]; }
        values[j] = column[index_pos::<8>(idx1, ROWS)];
    }
    let mut scaled = [0; COLS];
    for j in 0..COLS { scaled[j] = div(values[j], divide_number); }
    for i in 0..ROWS { for j in 0..COLS {
        a[i*COLS+j] = if hits1[i] { scaled[j] } else { a[i*COLS+j] };
    }}

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
