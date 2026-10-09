//! pythran_lu__factorMatrix0 (Pythran lu), halo2-base baseline: literal
//! translation of kernel.py (LITERAL_BASELINES.md).
//!
//! Kernel (factorMatrix0): LU elimination with partial pivoting of
//! `M: Float[N, COLS]` in place; the output is `M` after the call
//! (`mutated_outputs`). For each `line < N - 1`:
//!   cmax = np.argmax(abs(M[line:n, line])) + line
//!   if cmax != line: M[[line, cmax]] = M[[cmax, line]]
//!   pivot = M[line, line]
//!   for j in range(line + 1, n):
//!       v = M[j, line] / pivot
//!       for k in range(line, m): M[j, k] -= v * M[line, k]
//!
//! Interpretations (L1-L6):
//! - `np.argmax(abs(...))` (L5): a linear scan over rows `line..N` of column
//!   `line`, tracking the best magnitude and its 0-based position in the
//!   subarray as plain values (not a one-hot), with a strict `>` so the
//!   first maximum wins; `cmax = argmax + line` is then one addition, as in
//!   the source.
//! - `if cmax != line: M[[line, cmax]] = M[[cmax, line]]` (L4/L5): the
//!   comparison `cmax != line` is evaluated explicitly as its own step. The
//!   swap reads row `cmax` by the private 0-based position with `Q32::index`
//!   (L5: "a[i] with private index: library's standard indexing gadget --
//!   this IS the direct form") and writes it back with the private-index
//!   write formula `a[k] = select(i == k, v, a[k])`, both gated by that
//!   comparison -- a separate step from the argmax scan, not fused with it.
//! - `v = M[j, line] / pivot` (L5): the plain `Q32::div`, which accepts a
//!   divisor of either sign. The optimized baseline's `div_by_abs` reuses
//!   the argmax scan's running magnitude (`largest`) as the pivot's
//!   magnitude to skip `div`'s own sign test and window check; L6 forbids
//!   both `div_by_abs` and reusing an `I64`/magnitude from one source
//!   operation (the argmax) in another (the division), so this baseline
//!   takes a fresh `Q32::bound` of the pivot and calls plain `Q32::div`.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.factorMatrix0.json.
const N: usize = 5; // = M_0
const COLS: usize = 5; // = M_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m_in = inputs.floats("M", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut m = q.load_floats(ctx, &m_in);
    let at = |i: usize, k: usize| i * COLS + k;
    for line in 0..N.saturating_sub(1) {
        // cmax = np.argmax(abs(M[line:n, line])) + line.
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

        // if cmax != line: M[[line, cmax]] = M[[cmax, line]]
        let same = q.eq(ctx, cmax, line_const);
        let swap = q.gate.not(ctx, same);
        for k in 0..COLS {
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
