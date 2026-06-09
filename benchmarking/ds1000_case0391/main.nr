// DS-1000 case 0391 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[:, low:high] on Float[20,10].

global ROWS: u32 = 20;
global COLS: u32 = 10;

fn main(a: [Field; 200], low: Field, high: Field, claimed: [Field; 80]) -> pub [Field; 200] {
    assert(((low as u128) <= (high as u128)), "0391: low must be <= high");
    assert(((high as u128) <= (COLS as u128)), "0391: high must be <= COLS");
    let _ = claimed[0];
    let width: Field = high - low;
    let mut out: [Field; 200] = [0; 200];
    for i in 0..ROWS {
        for k in 0..COLS {
            if ((k as u128) < (width as u128)) {
                let mut v: Field = 0;
                for src_off in 0..COLS {
                    if (src_off as Field) == low + (k as Field) {
                        v = a[i * COLS + src_off];
                    }
                }
                out[i * COLS + k] = v;
            }
        }
    }
    out
}
