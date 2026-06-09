// DS-1000 case 0363 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// np.insert(a, pos, element) on Float[100] -> Float[101]. element promoted to Q.

use noir_fpc::Q;

global N: u32 = 100;

fn main(a: [Field; 100], pos: Field, element: Field) -> pub [Field; 101] {
    assert(((pos as u128) <= (N as u128)), "0363: pos out of range");
    let elem_fp = element * Q;
    let mut out: [Field; 101] = [0; 101];
    // For each output cell d, decide its source: if d < pos use a[d];
    // if d == pos use element; if d > pos use a[d-1].
    for d in 0..(N + 1) {
        let df: Field = d as Field;
        let lt_pos: Field = if (df as u128) < (pos as u128) { 1 } else { 0 };
        let eq_pos: Field = if df == pos { 1 } else { 0 };
        let gt_pos: Field = 1 - lt_pos - eq_pos;
        // Source for `lt`: a[d]; for `gt`: a[d-1].
        let a_d: Field = if d < N { a[d] } else { 0 };
        let a_dm1: Field = if d > 0 { if (d - 1) < N { a[d - 1] } else { 0 } } else { 0 };
        out[d] = lt_pos * a_d + eq_pos * elem_fp + gt_pos * a_dm1;
    }
    out
}
