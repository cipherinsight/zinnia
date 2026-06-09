// DS-1000 case 0364 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// np.insert(a, pos, element, axis=0) on Float[100,10] -> Float[101,10].

global ROWS: u32 = 100;
global COLS: u32 = 10;

fn main(a: [Field; 1000], pos: Field, element: [Field; 10]) -> pub [Field; 1010] {
    assert(((pos as u128) <= (ROWS as u128)), "0364: pos out of range");
    let mut out: [Field; 1010] = [0; 1010];
    for d in 0..(ROWS + 1) {
        let df: Field = d as Field;
        let lt_pos: Field = if (df as u128) < (pos as u128) { 1 } else { 0 };
        let eq_pos: Field = if df == pos { 1 } else { 0 };
        let gt_pos: Field = 1 - lt_pos - eq_pos;
        for j in 0..COLS {
            let a_d: Field = if d < ROWS { a[d * COLS + j] } else { 0 };
            let a_dm1: Field = if d > 0 { if (d - 1) < ROWS { a[(d - 1) * COLS + j] } else { 0 } } else { 0 };
            out[d * COLS + j] = lt_pos * a_d + eq_pos * element[j] + gt_pos * a_dm1;
        }
    }
    out
}
