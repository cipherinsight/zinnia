// DS-1000 case 0364 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// np.insert(a, pos, element, axis=0) on Float[100,10] -> Float[101,10].

use noir_fpc::is_negative;

global ROWS: u32 = 100;
global COLS: u32 = 10;

fn main(a: [Field; 1000], pos: Field, element: [Field; 10]) -> pub [Field; 1010] {
    // Full numpy semantics: negative positions wrap (np.insert with
    // pos = -1 inserts before the last row). One signed test, then all
    // per-slot comparisons run on the normalized (nonnegative) value;
    // the u128-cast assert enforces -ROWS <= pos <= ROWS.
    let neg = is_negative(pos);
    let pos_n = if neg { pos + (ROWS as Field) } else { pos };
    assert(((pos_n as u128) <= (ROWS as u128)), "0364: pos out of range");
    let mut out: [Field; 1010] = [0; 1010];
    for d in 0..(ROWS + 1) {
        let df: Field = d as Field;
        let lt_pos: Field = if (df as u128) < (pos_n as u128) { 1 } else { 0 };
        let eq_pos: Field = if df == pos_n { 1 } else { 0 };
        let gt_pos: Field = 1 - lt_pos - eq_pos;
        for j in 0..COLS {
            let a_d: Field = if d < ROWS { a[d * COLS + j] } else { 0 };
            let a_dm1: Field = if d > 0 { if (d - 1) < ROWS { a[(d - 1) * COLS + j] } else { 0 } } else { 0 };
            out[d * COLS + j] = lt_pos * a_d + eq_pos * element[j] + gt_pos * a_dm1;
        }
    }
    out
}
