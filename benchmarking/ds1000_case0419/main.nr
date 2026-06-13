// DS-1000 case 0419 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// data[:, ::-1] then bin-mean axis=-1 then [::-1] on Float[8,6].

use noir_fpc::{qadd, qdiv, Q, q_floor_div_bounded};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(data: [Field; 48], bin_size: Field) -> pub [Field; 48] {
    assert(bin_size != 0, "0419: bin_size must be > 0");
    // Sound floor-div (operands fit in 2^4 > 6): pins n_bins to the true floor
    // and rejects the field-inverse cheat `n_bins*bin_size == COLS` alone admits.
    let n_bins = q_floor_div_bounded::<4>(COLS as Field, bin_size);
    assert(n_bins * bin_size == COLS as Field, "bin_size must divide COLS");
    let bs_q: Field = bin_size * Q;
    let mut out: [Field; 48] = [0; 48];
    for i in 0..ROWS {
        for bin in 0..COLS {
            if ((bin as u128) < (n_bins as u128)) {
                let mut sum: Field = 0;
                for j in 0..COLS {
                    if ((j as u128) < (bin_size as u128)) {
                        let col_idx = (COLS - 1) as Field - ((bin as Field) * bin_size + (j as Field));
                        let src_idx = (i * COLS) as Field + col_idx;
                        let mut v: Field = 0;
                        for k in 0..(ROWS * COLS) {
                            if (k as Field) == src_idx { v = data[k]; }
                        }
                        sum = qadd(sum, v);
                    }
                }
                let val = qdiv(sum, bs_q);
                for out_pos in 0..COLS {
                    let in_range = (out_pos as u128) < (n_bins as u128);
                    let target = (out_pos as Field) + (bin as Field) + 1 == n_bins;
                    if in_range & target {
                        out[i * COLS + out_pos] = val;
                    }
                }
            }
        }
    }
    out
}
