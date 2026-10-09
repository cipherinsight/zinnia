// DS-1000 case 0346, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel:
//   a[row-1, :] *= multiply_number
//   result = np.cumsum(a[row-1, :])
// on a: Float[8, 6], row: Integer, multiply_number: Float. The row analog of
// case 0345: NumPy index semantics require row - 1 in [-8, 8) (negative
// indices count from the end); any other value has no witness. row - 1 is
// written twice in the source, so it is computed, and its validity checked,
// twice (no CSE across statements). There is no shared gadget for "write at
// a private index" (only reads), so the write uses the local helper
// `row_hits` below, per the L5 rule for a[i] = v. The write touches every
// one of the 48 cells of a (L2: the full mutated array), selecting per row
// and column whether that row is the target row. The read (a[row-1, :]) is
// the direct form of an indexed read (noir_hw::index_pos), done once per
// column (each column is an 8-element array indexed by row - 1). np.cumsum
// is a running accumulator over the 6 columns in order (L5).
// @outputs: float[6]

use noir_hw::{index_pos, mul};

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

fn main(a: [Field; 48], row: Field, multiply_number: Field) -> pub [Field; 6] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_input(row);
    noir_hw::check_input(multiply_number);

    // a[row-1, :] *= multiply_number
    let idx1 = row - 1;
    let hits1 = row_hits(idx1, ROWS);
    let mut a = a;
    for i in 0..ROWS {
        for j in 0..COLS {
            let old = a[i * COLS + j];
            let scaled = mul(old, multiply_number);
            a[i * COLS + j] = if hits1[i] { scaled } else { old };
        }
    }

    // result = np.cumsum(a[row-1, :])
    let idx2 = row - 1; // recomputed: no CSE across statements
    let mut cumsum: [Field; 6] = [0; COLS];
    let mut acc: Field = 0;
    for j in 0..COLS {
        let mut column: [Field; 8] = [0; ROWS];
        for i in 0..ROWS {
            column[i] = a[i * COLS + j];
        }
        let picked = column[index_pos::<8>(idx2, ROWS)];
        acc += picked;
        cumsum[j] = acc;
    }
    cumsum
}
