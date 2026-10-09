// pythran_make_decision__md: literal translation of kernel.py (LITERAL_BASELINES.md).
// Complex abs follows the current optimized Q32 schedule (rounded component squares, sqrt). Full arrays, first-index argmin scan, then standard gather.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const L: usize = 16; // = E_0
const M: usize = 8; // = symbols_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let e = inputs.complexes("E", &[L])?;
    let symbols = inputs.complexes("symbols", &[M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let e = q.load_floats(ctx, &e);
    let symbols = q.load_floats(ctx, &symbols);
    let zero = q.float_const(ctx, 0.0);
    let mut result = vec![zero; 2 * L];
    for i in 0..L {
        let difference: Vec<[Cell; 2]> = (0..M)
            .map(|k| {
                [
                    q.sub(ctx, e[2 * i], symbols[2 * k]),
                    q.sub(ctx, e[2 * i + 1], symbols[2 * k + 1]),
                ]
            })
            .collect();
        let magnitude: Vec<Cell> = difference
            .iter()
            .map(|v| {
                let rr = q.mul(ctx, v[0], v[0]).cell;
                let ii = q.mul(ctx, v[1], v[1]).cell;
                let sum = q.add(ctx, rr, ii);
                let sum = q.bound(ctx, sum).cell;
                q.sqrt(ctx, sum).cell
            })
            .collect();
        let distances: Vec<Cell> = magnitude.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
        let mut best = distances[0];
        let mut im = q.int_const(ctx, 0);
        for k in 1..M {
            let a = q.bound(ctx, distances[k]);
            let b = q.bound(ctx, best);
            let better = q.lt(ctx, a, b);
            best = q.gate.select(ctx, distances[k], best, better);
            im = q.gate.select(ctx, Constant(fe(k as i128)), im, better);
        }
        let re: Vec<Cell> = (0..M).map(|k| symbols[2 * k]).collect();
        let imaginary: Vec<Cell> = (0..M).map(|k| symbols[2 * k + 1]).collect();
        result[2 * i] = q.index(ctx, &re, im);
        result[2 * i + 1] = q.index(ctx, &imaginary, im);
    }
    out.floats(&result);
    Ok(())
}
