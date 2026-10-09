//! pythran_lu__factorMatrix0 (Pythran lu), halo2-base baseline.
//! Kernel: LU elimination with partial pivoting of `M: Float[N, COLS]` in
//! place; the output is `M` after the call (`mutated_outputs`). For each
//! `line < N - 1`: the pivot row is the first row `i >= line` with the
//! largest `|M[i, line]|` (`np.argmax`), it is swapped with row `line`, and
//! every row below subtracts `M[j, line] / pivot` times row `line` from
//! column `line` on.
//! The pivot search is a chain of 63-bit comparisons of the magnitudes
//! (`abs_int64`), keeping a one-hot of the row reached so far (a strict `<`
//! keeps the first maximum); the swap selects whole rows through the one-hot.
//! The largest magnitude is the pivot's, so the quotients divide by it
//! directly (`div_by_abs`: no sign test, no window check of the pivot).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.factorMatrix0.json.
const N: usize = 5; // = M_0
const COLS: usize = 5; // = M_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m = inputs.floats("M", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut m = q.load_floats(ctx, &m);
    let at = |i: usize, k: usize| i * COLS + k;
    for line in 0..N.saturating_sub(1) {
        let magnitudes: Vec<Cell> = (line..N).map(|i| q.abs_int64(ctx, m[at(i, line)])).collect();
        let (hot, largest) = first_argmax(&q, ctx, &magnitudes);
        swap_rows(&q, ctx, &mut m, line, &hot);
        let pivot = m[at(line, line)];
        for j in line + 1..N {
            let v = q.div_by_abs(ctx, m[at(j, line)], pivot, largest);
            for k in line..COLS {
                let product = q.mul(ctx, v, m[at(line, k)]);
                m[at(j, k)] = q.sub(ctx, m[at(j, k)], product);
            }
        }
    }
    out.floats(&m);
    Ok(())
}

/// The one-hot of the first maximum of `magnitudes` (each proven below
/// `2^63`) and the maximum itself.
fn first_argmax(q: &Q32, ctx: &mut Ctx, magnitudes: &[Cell]) -> (Vec<Cell>, Cell) {
    let mut largest = magnitudes[0];
    let mut hot = vec![ctx.load_constant(Fr::ONE)];
    for &candidate in &magnitudes[1..] {
        let larger = q.range.is_less_than(ctx, largest, candidate, 63);
        largest = q.gate.select(ctx, candidate, largest, larger);
        for h in hot.iter_mut() {
            *h = q.gate.sub_mul(ctx, *h, *h, larger); // h * (1 - larger)
        }
        hot.push(larger);
    }
    (hot, largest)
}

/// `M[[line, cmax]] = M[[cmax, line]]` for the row `cmax = line + t` marked
/// by `hot[t]` (no change when `cmax == line`).
fn swap_rows(q: &Q32, ctx: &mut Ctx, m: &mut [Cell], line: usize, hot: &[Cell]) {
    for k in 0..COLS {
        let column: Vec<Cell> = (line..N).map(|i| m[i * COLS + k]).collect();
        let top = column[0];
        m[line * COLS + k] =
            q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), column.iter().map(|c| Existing(*c)));
        for t in 1..column.len() {
            let gap = q.sub(ctx, top, column[t]);
            m[(line + t) * COLS + k] = q.gate.mul_add(ctx, hot[t], gap, column[t]);
        }
    }
}
