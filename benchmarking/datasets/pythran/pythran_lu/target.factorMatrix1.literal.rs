//! pythran_lu__factorMatrix1 (Pythran lu), halo2-base baseline: literal
//! translation of kernel.py (LITERAL_BASELINES.md).
//!
//! Kernel (factorMatrix1): LU elimination with partial pivoting of
//! `M: Float[N, COLS]` in place; the output is `M` after the call
//! (`mutated_outputs`). For each `line < N - 1`:
//!   cmax = line; vmax = abs(M[line, line])
//!   for i in range(line + 1, n):
//!       if abs(M[i, line]) > vmax: vmax = abs(M[i, line]); cmax = i
//!   if cmax != line:
//!       for j in range(line, m): t = M[line,j]; M[line,j] = M[cmax,j]; M[cmax,j] = t
//!   pivot = M[line, line]
//!   for j in range(line + 1, n):
//!       v = M[j, line] / pivot
//!       for k in range(line, m): M[j, k] -= v * M[line, k]
//!
//! Interpretations (L1-L6): same as `factorMatrix0`'s literal baseline --
//! the `if ... > vmax: ...; cmax = i` loop body (L4) is the same running
//! best-value/best-index update as an argmax scan; the explicit
//! column-by-column swap with a temporary (`t = M[line,j]; ...`) is the
//! same private-index row swap, restricted to columns `line..m` as the
//! source loop says (unlike `factorMatrix0`'s fancy-index swap, which moves
//! the whole row); and `v = M[j, line] / pivot` uses the plain `Q32::div`
//! rather than the optimized baseline's `div_by_abs` reuse of the argmax
//! scan's magnitude (L6-forbidden cross-statement reuse).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.factorMatrix1.json.
const N: usize = 5; // = M_0
const COLS: usize = 5; // = M_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m_in = inputs.floats("M", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut m = q.load_floats(ctx, &m_in);
    let at = |i: usize, k: usize| i * COLS + k;
    for line in 0..N.saturating_sub(1) {
        // cmax = line; vmax = abs(M[line, line]).
        // for i in line+1..n: if abs(M[i, line]) > vmax: vmax = ...; cmax = i.
        let mut best_val = q.abs_int64(ctx, m[at(line, line)]);
        let mut rel_idx = ctx.load_zero();
        for (step, i) in (line + 1..N).enumerate() {
            let candidate = q.abs_int64(ctx, m[at(i, line)]);
            let larger = q.range.is_less_than(ctx, best_val, candidate, 63);
            best_val = q.gate.select(ctx, candidate, best_val, larger);
            let step_const = ctx.load_constant(Fr::from((step + 1) as u64));
            rel_idx = q.gate.select(ctx, step_const, rel_idx, larger);
        }
        let line_const = ctx.load_constant(Fr::from(line as u64));
        let cmax = q.gate.add(ctx, rel_idx, line_const);

        // if cmax != line: for j in line..m: swap M[line, j] and M[cmax, j].
        let same = q.eq(ctx, cmax, line_const);
        let swap = q.gate.not(ctx, same);
        for k in line..COLS {
            let row_line_k = m[at(line, k)];
            let candidates: Vec<Cell> = (line..N).map(|p| m[at(p, k)]).collect();
            let row_cmax_k = q.index(ctx, &candidates, rel_idx);
            m[at(line, k)] = q.gate.select(ctx, row_cmax_k, row_line_k, swap);
            for p in line + 1..N {
                let pos_const = ctx.load_constant(Fr::from((p - line) as u64));
                let is_cmax = q.eq(ctx, rel_idx, pos_const);
                let hit = q.gate.mul(ctx, is_cmax, swap);
                m[at(p, k)] = q.gate.select(ctx, row_line_k, m[at(p, k)], hit);
            }
        }

        // pivot = M[line, line]; for j in line+1..n: v = M[j, line] / pivot.
        let pivot = q.bound(ctx, m[at(line, line)]);
        for j in line + 1..N {
            let v = q.div(ctx, m[at(j, line)], pivot);
            for k in line..COLS {
                let product = q.mul(ctx, v, m[at(line, k)]);
                m[at(j, k)] = q.sub(ctx, m[at(j, k)], product);
            }
        }
    }
    out.floats(&m);
    Ok(())
}
