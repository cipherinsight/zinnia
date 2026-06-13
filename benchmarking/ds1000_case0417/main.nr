// DS-1000 case 0417 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// bin-mean on data[::-1] from Float[8] with runtime bin_size.

use noir_fpc::{qadd, qdiv, Q, q_floor_div_bounded};

global N: u32 = 8;

fn main(data: [Field; 8], bin_size: Field) -> pub [Field; 8] {
    assert(bin_size != 0, "0417: bin_size must be > 0");
    // Sound floor-div (operands fit in 2^5 > 8): pins n_bins to the true floor
    // and rejects the field-inverse cheat `n_bins*bin_size == N` alone admits.
    let n_bins = q_floor_div_bounded::<5>(N as Field, bin_size);
    assert(n_bins * bin_size == N as Field, "bin_size must divide N");
    let bs_q: Field = bin_size * Q;
    let mut out: [Field; 8] = [0; 8];
    for bin in 0..N {
        if ((bin as u128) < (n_bins as u128)) {
            let mut sum: Field = 0;
            for j in 0..N {
                if ((j as u128) < (bin_size as u128)) {
                    let src_idx = (N - 1) as Field - ((bin as Field) * bin_size + (j as Field));
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
