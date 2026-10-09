// pythran_make_decision__md (Pythran make_decision), Noir baseline.
// Kernel: for each E[i] of E: Complex[L], the symbol of symbols: Complex[M]
// nearest to it, symbols[np.argmin(abs(E[i] - symbols) ** 2)] (the first one
// on ties); returns them as Complex[L].
// The comparisons are on the exact squared distances
// R_k = (Re E - Re s_k)^2 + (Im E - Im s_k)^2 (raw, scale 2^64), whose order
// is that of the real abs(E[i] - s_k) ** 2: no square root, no rounding. The
// argmin is a one-hot hint b (bits, one of them set) pinned by
// R_k - m - [k before the hinted index] in [0, 2^GAP_BITS) for every k, with
// m = sum_k b_k R_k: the hinted index holds the first minimum. 98 bits (seven
// 14-bit limbs) admit every squared distance below 2^34 (float), so each
// squared distance whose Q32 value fits the window; a larger one, an
// intermediate the window cannot hold, has no proof. The symbol is read
// through the one-hot.
// @outputs: float[2 * L]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.json.
global L: u32 = 16; // = E_0
global M: u32 = 8; // = symbols_0

global GAP_BITS: u32 = 98;

// The one-hot vector of the first minimum of the non-negative integers r.
unconstrained fn argmin_hint(r: [Field; M]) -> [Field; M] {
    let mut best = 0;
    for k in 1..M {
        if r[k].lt(r[best]) {
            best = k;
        }
    }
    let mut hot: [Field; M] = [0; M];
    hot[best] = 1;
    hot
}

fn main(E: [[Field; 2]; L], symbols: [[Field; 2]; M]) -> pub [[Field; 2]; L] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..L {
        check_inputs(E[i]);
    }
    for k in 0..M {
        check_inputs(symbols[k]);
    }
    let mut result: [[Field; 2]; L] = [[0; 2]; L];
    for i in 0..L {
        let mut r: [Field; M] = [0; M];
        for k in 0..M {
            let dr = E[i][0] - symbols[k][0];
            let di = E[i][1] - symbols[k][1];
            r[k] = dr * dr + di * di;
        }
        // Safety: pinned by the bit, count and gap checks below.
        let hot = unsafe { argmin_hint(r) };
        let mut count: Field = 0;
        let mut minimum: Field = 0;
        for k in 0..M {
            assert(hot[k] * (hot[k] - 1) == 0, "argmin: not a bit");
            count += hot[k];
            minimum += hot[k] * r[k];
        }
        assert(count == 1, "argmin: not one-hot");
        // `after` = [the hinted index is after k].
        let mut after: Field = 0;
        for t in 0..M {
            let k = M - 1 - t;
            (r[k] - minimum - after).assert_max_bit_size::<GAP_BITS>();
            after += hot[k];
        }
        for part in 0..2 {
            let mut picked: Field = 0;
            for k in 0..M {
                picked += hot[k] * symbols[k][part];
            }
            result[i][part] = picked;
        }
    }
    result
}
