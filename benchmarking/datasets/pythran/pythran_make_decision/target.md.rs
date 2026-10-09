//! pythran_make_decision__md (Pythran make_decision), halo2-base baseline.
//! Kernel: for each `E[i]` of E: Complex[L], the symbol of symbols: Complex[M]
//! nearest to it, `symbols[np.argmin(abs(E[i] - symbols) ** 2)]` (the first
//! one on ties); returns them as Complex[L].
//! The comparisons are on the exact squared distances
//! `R_k = (Re E - Re s_k)^2 + (Im E - Im s_k)^2` (raw, scale 2^64), whose
//! order is that of the real `abs(E[i] - s_k) ** 2`: no square root, no
//! rounding. The argmin is a one-hot hint `b` (bits, one of them set) pinned by
//! `R_k - m - [k before the hinted index]` in `[0, 2^96)` for every `k`, with
//! `m = sum_k b_k R_k`: the hinted index holds the first minimum. 96 bits
//! admit every squared distance below 2^32 (float), so each squared distance
//! whose Q32 value fits the window; a larger one, an intermediate the window
//! cannot hold, has no proof. The symbol is read through the one-hot.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const L: usize = 16; // = E_0
const M: usize = 8; // = symbols_0

/// Width of the gap checks: squared distances below 2^32 (float) fit.
const GAP_BITS: usize = 96;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let e = inputs.complexes("E", &[L])?;
    let symbols = inputs.complexes("symbols", &[M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let e = q.load_floats(ctx, &e); // interleaved real and imaginary parts
    let symbols = q.load_floats(ctx, &symbols);
    let (re, im): (Vec<Cell>, Vec<Cell>) = symbols.chunks(2).map(|s| (s[0], s[1])).unzip();
    let mut result = Vec::with_capacity(2 * L);
    for i in 0..L {
        let distances: Vec<Cell> = (0..M)
            .map(|k| {
                let dr = q.sub(ctx, e[2 * i], re[k]);
                let di = q.sub(ctx, e[2 * i + 1], im[k]);
                q.gate.inner_product(ctx, [Existing(dr), Existing(di)], [Existing(dr), Existing(di)])
            })
            .collect();
        let hot = first_argmin(&q, ctx, &distances);
        let pick = |ctx: &mut Ctx, values: &[Cell]| {
            q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), values.iter().map(|v| Existing(*v)))
        };
        result.push(pick(ctx, &re));
        result.push(pick(ctx, &im));
    }
    out.floats(&result);
    Ok(())
}

/// The one-hot vector of the first minimum of the non-negative integers `r`,
/// pinned as described in the header.
fn first_argmin(q: &Q32, ctx: &mut Ctx, r: &[Cell]) -> Vec<Cell> {
    let values: Vec<_> = r.iter().map(|c| halo2_base::utils::fe_to_biguint(c.value())).collect();
    let first = (0..r.len()).fold(0, |best, k| if values[k] < values[best] { k } else { best });
    let hot: Vec<Cell> = (0..r.len()).map(|k| ctx.load_witness(Fr::from((k == first) as u64))).collect();
    for h in &hot {
        q.gate.assert_bit(ctx, *h);
    }
    let count = q.sum(ctx, hot.iter().copied());
    q.gate.assert_is_const(ctx, &count, &Fr::ONE);
    let minimum = q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), r.iter().map(|v| Existing(*v)));
    // after = [the hinted index is after k], a suffix sum of the one-hot.
    let mut after: Option<Cell> = None;
    for k in (0..r.len()).rev() {
        let gap = q.sub(ctx, r[k], minimum);
        let gap = match after {
            Some(a) => q.sub(ctx, gap, a),
            None => gap,
        };
        q.range.range_check(ctx, gap, GAP_BITS);
        after = Some(match after {
            Some(a) => q.add(ctx, a, hot[k]),
            None => hot[k],
        });
    }
    hot
}
