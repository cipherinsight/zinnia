//! pythran_make_decision__md (Pythran make_decision), halo2-base baseline.
//! Kernel: for each `E[i]` of E: Complex[L], the symbol of symbols: Complex[M]
//! nearest to it, `symbols[np.argmin(abs(E[i] - symbols) ** 2)]` (the first
//! one on ties); returns them as Complex[L].
//! Preserve the source's complex magnitude, then square, then first argmin.
//! Each component square is Q32-rounded before the sum and square root;
//! the magnitude is squared with another Q32 rounding. Compare sequentially
//! with strict less-than to preserve first-index ties. No argmin hint.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const L: usize = 16; // = E_0
const M: usize = 8; // = symbols_0

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
        let distances: Vec<I64> = (0..M)
            .map(|k| {
                let dr = q.sub(ctx, e[2 * i], re[k]);
                let di = q.sub(ctx, e[2 * i + 1], im[k]);
                let rr = q.mul(ctx, dr, dr).cell;
                let ii = q.mul(ctx, di, di).cell;
                let norm2 = q.add(ctx, rr, ii);
                // Frozen Halo2 sqrt falls back to signed64 sign handling here.
                let norm2 = q.bound(ctx, norm2).cell;
                let magnitude = q.sqrt(ctx, norm2).cell;
                q.mul(ctx, magnitude, magnitude)
            })
            .collect();
        let mut minimum = distances[0];
        let mut selected_re = re[0];
        let mut selected_im = im[0];
        for k in 1..M {
            let better = q.lt(ctx, distances[k], minimum);
            minimum = q.select_i64(ctx, distances[k], minimum, better);
            selected_re = q.gate.select(ctx, re[k], selected_re, better);
            selected_im = q.gate.select(ctx, im[k], selected_im, better);
        }
        result.push(selected_re);
        result.push(selected_im);
    }
    out.floats(&result);
    Ok(())
}
