// DS-1000 case 0434 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[zero_rows, :] = 0; a[:, zero_cols] = 0 on Integer[4,4].

global ROWS: u32 = 4;
global COLS: u32 = 4;

fn main(a: [Field; 16], zero_rows: [Field; 2], zero_cols: [Field; 2]) -> pub [Field; 16] {
    let mut out: [Field; 16] = [0; 16];
    for i in 0..ROWS {
        for j in 0..COLS {
            let mut zero_it: bool = false;
            for r in 0..2 {
                if ((i as Field) == zero_rows[r])
                    | (((i as Field) - (ROWS as Field)) == zero_rows[r]) { zero_it = true; }
            }
            for c in 0..2 {
                if ((j as Field) == zero_cols[c])
                    | (((j as Field) - (COLS as Field)) == zero_cols[c]) { zero_it = true; }
            }
            out[i * COLS + j] = if zero_it { 0 } else { a[i * COLS + j] };
        }
    }
    out
}
