// DS-1000 case 0345, Noir baseline, literal translation of kernel.py
// (LITERAL_BASELINES.md).
// Kernel:
//   a[:, col-1] *= multiply_number
//   result = np.cumsum(a[:, col-1])
// on a: Float[8, 6], col: Integer, multiply_number: Float. NumPy index
// semantics: col - 1 must lie in [-6, 6) (negative indices count from the
// end); any other value has no witness. col - 1 is written twice in the
// source (once per statement), so it is computed, and its validity checked,
// twice (no CSE across statements). There is no shared gadget for "write at
// a private index" (only reads), so the write uses the local helper
// `column_hits` below, per the L5 rule for a[i] = v. The write touches every
// one of the 48 cells of a (L2: the full mutated array), selecting per row
// and column whether that column is the target column. The read
// (a[:, col-1]) is the direct form of an indexed read (noir_hw::index_pos).
// np.cumsum is a running accumulator over the 8 rows in order (L5).
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
    for i in 0..ROWS {
        for k in 0..COLS {
            let old = a[i * COLS + k];
            let scaled = mul(old, multiply_number);
            a[i * COLS + k] = if hits1[k] { scaled } else { old };
        }
    }

    // result = np.cumsum(a[:, col-1])
    let idx2 = col - 1; // recomputed: no CSE across statements
    let mut cumsum: [Field; 8] = [0; ROWS];
    let mut acc: Field = 0;
    for i in 0..ROWS {
        let mut row: [Field; 6] = [0; COLS];
        for k in 0..COLS {
            row[k] = a[i * COLS + k];
        }
        let picked = row[index_pos::<8>(idx2, COLS)];
        acc += picked;
        cumsum[i] = acc;
    }
    cumsum
}
