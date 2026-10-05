// DS-1000 case 0392 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[low:high, :] on Float[20,10].

global ROWS: u32 = 20;
global COLS: u32 = 10;

fn main(a: [Field; 200], low: Field, high: Field, claimed: [Field; 100]) -> pub [Field; 200] {
    assert(((low as u128) <= (high as u128)), "0392: low must be <= high");
    assert(((high as u128) <= (ROWS as u128)), "0392: high must be <= ROWS");
    let _ = claimed[0];
    let nrows: Field = high - low;
    let mut out: [Field; 200] = [0; 200];
    for i in 0..ROWS {
        if ((i as u128) < (nrows as u128)) {
            for j in 0..COLS {
                let mut v: Field = 0;
                for src_row in 0..ROWS {
                    if (src_row as Field) == low + (i as Field) {
                        v = a[src_row * COLS + j];
                    }
                }
                out[i * COLS + j] = v;
            }
        }
    }
    out
}
