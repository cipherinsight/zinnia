//! pythran_lu__factorMatrix1: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Repeat abs in the taken branch; private writes scan every row, then select the branch.
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
        // cmax = line; vmax = abs(M[line,line]); scan with conditional updates.
        let line_const = q.int_const(ctx, line as i64);
        let mut cmax = line_const;
        let mut best_val = q.abs_int64(ctx, m[at(line, line)]);
        for i in line + 1..N {
            let candidate = q.abs_int64(ctx, m[at(i, line)]);
            let a = q.bound(ctx, candidate);
            let b = q.bound(ctx, best_val);
            let larger = q.gt(ctx, a, b);
            let assigned = q.abs_int64(ctx, m[at(i, line)]);
            best_val = q.gate.select(ctx, assigned, best_val, larger);
            let index = q.int_const(ctx, i as i64);
            cmax = q.gate.select(ctx, index, cmax, larger);
        }
        let same = q.eq(ctx, cmax, line_const);
        let swap = q.gate.not(ctx, same);
        let old = m.clone();
        let mut branch = old.clone();
        for k in line..COLS {
            let t = branch[at(line, k)];
            let column: Vec<Cell> = (0..N).map(|p| branch[at(p, k)]).collect();
            branch[at(line, k)] = q.index(ctx, &column, cmax);
            for p in 0..N {
                let hit = q.eq(ctx, cmax, Constant(fe(p as i128)));
                branch[at(p, k)] = q.gate.select(ctx, t, branch[at(p, k)], hit);
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
