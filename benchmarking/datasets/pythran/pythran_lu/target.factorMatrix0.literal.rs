//! pythran_lu__factorMatrix0: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize abs, then scan argmax; gather both RHS rows before scatter/select.
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
        let magnitudes: Vec<Cell> = (line..N)
            .map(|i| q.abs_int64(ctx, m[at(i, line)]))
            .collect();
        let mut best_val = magnitudes[0];
        let mut rel_idx = ctx.load_zero();
        for step in 1..magnitudes.len() {
            let candidate = magnitudes[step];
            let a = q.bound(ctx, candidate);
            let b = q.bound(ctx, best_val);
            let larger = q.gt(ctx, a, b);
            best_val = q.gate.select(ctx, candidate, best_val, larger);
            let index = q.int_const(ctx, step as i64);
            rel_idx = q.gate.select(ctx, index, rel_idx, larger);
        }
        let line_const = q.int_const(ctx, line as i64);
        let cmax = q.add(ctx, rel_idx, line_const);
        let same = q.eq(ctx, cmax, line_const);
        let swap = q.gate.not(ctx, same);
        let old = m.clone();
        let row: Vec<Cell> = (0..COLS)
            .map(|k| {
                let column: Vec<Cell> = (0..N).map(|p| old[at(p, k)]).collect();
                q.index(ctx, &column, cmax)
            })
            .collect();
        let mut branch = old.clone();
        for k in 0..COLS {
            branch[at(line, k)] = row[k];
        }
        for p in 0..N {
            for k in 0..COLS {
                let hit = q.eq(ctx, cmax, Constant(fe(p as i128)));
                branch[at(p, k)] = q.gate.select(ctx, old[at(line, k)], branch[at(p, k)], hit);
            }
        }
        for p in 0..N * COLS {
            m[p] = q.gate.select(ctx, branch[p], old[p], swap);
        }

        // pivot = M[line, line]; for j in line+1..n: v = M[j, line] / pivot.
        let pivot = m[at(line, line)];
        for j in line + 1..N {
            let divisor = q.bound(ctx, pivot);
            let v = q.div(ctx, m[at(j, line)], divisor);
            for k in line..COLS {
                let product = q.mul(ctx, v, m[at(line, k)]);
                m[at(j, k)] = q.sub(ctx, m[at(j, k)], product);
            }
        }
    }
    out.floats(&m);
    Ok(())
}
