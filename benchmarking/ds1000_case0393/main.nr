// DS-1000 case 0393 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// high = min(high, a.shape[1]); a[:, low:high] on Float[20,10].

global ROWS: u32 = 20;
global COLS: u32 = 10;

fn main(a: [Field; 200], low: Field, high: Field, claimed: [Field; 140]) -> pub [Field; 200] {
    assert(((low as u128) <= (COLS as u128)), "0393: low must be in [0, COLS]");
    let _ = claimed[0];
    let high_eff: Field = if (high as u128) > (COLS as u128) { COLS as Field } else { high };
    assert(((low as u128) <= (high_eff as u128)), "0393: low must be <= effective high");
    let width: Field = high_eff - low;
    let mut out: [Field; 200] = [0; 200];
    for i in 0..ROWS {
        for k in 0..COLS {
            if ((k as u128) < (width as u128)) {
                let src_col = low + (k as Field);
                let mut v: Field = 0;
                for c in 0..COLS {
                    if (c as Field) == src_col { v = a[i * COLS + c]; }
                }
                out[i * COLS + k] = v;
            }
        }
    }
    out
}
