// DS-1000 case 0414 -- Noir baseline (faithful, runtime bin_size).
// Field convention (CONVENTIONS.md sec 3).
// bin-mean on Float[8] with runtime bin_size. Output max-bound [Field; 8]
// (when bin_size=1, n_bins=8). Cells beyond actual n_bins are zero.

use noir_fpc::{qadd, qdiv, Q, q_floor_div_hint};

global N: u32 = 8;

fn main(data: [Field; 8], bin_size: Field) -> pub [Field; 8] {
    assert(bin_size != 0, "0414: bin_size must be > 0");
    let n_bins = q_floor_div_hint(N as Field, bin_size);
    assert(n_bins * bin_size == N as Field, "bin_size must divide N");
    let bs_q: Field = bin_size * Q; // promote integer to Q-scale
    let mut out: [Field; 8] = [0; 8];
    for bin in 0..N {
        if ((bin as u128) < (n_bins as u128)) {
            let mut sum: Field = 0;
            for j in 0..N {
                if ((j as u128) < (bin_size as u128)) {
                    let src_idx = (bin as Field) * bin_size + (j as Field);
                    let mut v: Field = 0;
                    for k in 0..N {
                        if (k as Field) == src_idx { v = data[k]; }
                    }
                    sum = qadd(sum, v);
                }
            }
            out[bin] = qdiv(sum, bs_q);
        }
    }
    out
}
