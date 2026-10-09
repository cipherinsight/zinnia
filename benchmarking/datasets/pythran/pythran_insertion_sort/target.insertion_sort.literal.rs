// pythran_insertion_sort__insertion_sort: literal translation of kernel.py (LITERAL_BASELINES.md).
// While bounded by i; private j retained, active gates every effect. Reads use indexing; writes scan equality tests. At j=0 the inactive -1 read is valid.
use crate::prelude::*;
const N: usize = 10; // = list2_0
const INDEX_BITS: usize = 8; // = 8 + 20 * min(1, (list2_0 - 1) // 128)

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.floats("list2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut values = q.load_floats(ctx, &values);
    for i in 1..N {
        let save = values[i];
        let mut j = q.int_const(ctx, i as i64);
        let mut active = q.int_const(ctx, 1);
        for _step in 0..i {
            let jb = q.bound(ctx, j);
            let zero = q.bounded_const(ctx, 0);
            let positive = q.gt(ctx, jb, zero);
            let prev = q.sub(ctx, j, Constant(fe(1)));
            // At j=0, -1 is a valid NumPy read; the short-circuit flag masks its comparison.
            let compared = q.index(ctx, &values, prev);
            let cb = q.bound(ctx, compared);
            let sb = q.bound(ctx, save);
            let larger = q.gt(ctx, cb, sb);
            let guard = q.gate.and(ctx, positive, larger);
            active = q.gate.and(ctx, active, guard);
            // The RHS indexed read occurs again in the source body.
            let prev = q.sub(ctx, j, Constant(fe(1)));
            let shifted = q.index(ctx, &values, prev);
            for k in 0..N {
                let hit = q.eq(ctx, j, Constant(fe(k as i128)));
                let effect = q.gate.and(ctx, active, hit);
                values[k] = q.gate.select(ctx, shifted, values[k], effect);
            }
            let next = q.sub(ctx, j, Constant(fe(1)));
            j = q.gate.select(ctx, next, j, active);
        }
        for k in 0..N {
            let hit = q.eq(ctx, j, Constant(fe(k as i128)));
            values[k] = q.gate.select(ctx, save, values[k], hit);
        }
    }
    out.floats(&values);
    Ok(())
}
