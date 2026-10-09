// DS-1000 case 0345, literal translation of kernel.py (LITERAL_BASELINES.md).
// Gather the augmented-assignment slice, compute each element, then scatter by equality selects.
// Negative indices wrap; each source indexed expression is evaluated separately.
// Reduction is a running accumulator in index order.
// @outputs: float[8]
use noir_hw::{index_pos, mul};

global ROWS: u32 = 8;
global COLS: u32 = 6;

// The hit vector of a NumPy column index idx over len columns, for a
// private-index write (no shared gadget writes by a private index): column
// k is hit when idx is k or k - len (the wrapped negative index). Asserting
// exactly one hit pins idx to a valid NumPy index (any other value has no
// witness, as NumPy raises IndexError). Shared across the ROWS rows the
// write touches, since the source statement applies the one column index to
// every row.
fn column_hits(idx: Field, len: u32) -> [bool; COLS] {
    let mut hits: [bool; COLS] = [false; COLS];
    let mut total: Field = 0;
    for k in 0..len {
        let hit = (idx == k as Field) | (idx == k as Field - len as Field);
        hits[k] = hit;
        total += hit as Field;
    }
    assert(total == 1, "index out of bounds");
    hits
}

fn main(a: [Field; 48], col: Field, multiply_number: Field) -> pub [Field; 8] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(col);
    noir_hw::check_input(multiply_number);

    // a[:, col-1] *= multiply_number
    let idx1 = col - 1;
    let hits1 = column_hits(idx1, COLS);
    let mut a = a;
    let mut values = [0; ROWS];
    for i in 0..ROWS {
        let mut row_values = [0; COLS];
        for j in 0..COLS { row_values[j] = a[i*COLS+j]; }
        values[i] = row_values[index_pos::<8>(idx1, COLS)];
    }
    let mut scaled = [0; ROWS];
    for i in 0..ROWS { scaled[i] = mul(values[i], multiply_number); }
    for i in 0..ROWS { for k in 0..COLS {
        a[i*COLS+k] = if hits1[k] { scaled[i] } else { a[i*COLS+k] };
    }}

    // result = np.cumsum(a[:, col-1])
    let idx2 = col - 1; // recomputed: no CSE across statements
    let mut values = [0; ROWS];
    for i in 0..ROWS {
        let mut row_values = [0; COLS];
        for j in 0..COLS { row_values[j] = a[i*COLS+j]; }
        values[i] = row_values[index_pos::<8>(idx2, COLS)];
    }
    let mut cumsum = [0; ROWS];
    cumsum[0] = values[0];
    for i in 1..ROWS { cumsum[i] = cumsum[i-1] + values[i]; }
    cumsum
}
