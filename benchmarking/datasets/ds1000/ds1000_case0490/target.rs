//! DS-1000 case 0490, halo2-base baseline.
//! Kernel: for each row `a` of `arr: Float[5, 50]` with `t1 = n1[r]`,
//! `t2 = n2[r]` (`n1`, `n2: Integer[5]`): `temp = a.copy();
//! a[temp < t1] = 0; a[temp >= t2] = 30; a[(temp >= t1) & (temp < t2)] += 5`.
//! In assignment order a cell `x` ends as `30` if `x >= t2` (the second write
//! wins, also when `t2 < t1`), else `0` if `x < t1`, else `x + 5`: two
//! comparisons per cell, against the row's thresholds converted to floats once
//! per row (`t * 2^32` must lie in the window: `-2^31 <= t < 2^31`).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let arr = inputs.floats("arr", &[5, 50])?;
    let (n1, n2) = (inputs.ints("n1", &[5])?, inputs.ints("n2", &[5])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let arr = q.load_floats(ctx, &arr);
    let (n1, n2) = (q.load_ints(ctx, &n1), q.load_ints(ctx, &n2));
    let mut result = Vec::with_capacity(arr.len());
    for (row, (t1, t2)) in arr.chunks(50).zip(n1.into_iter().zip(n2)) {
        let t1 = q.int_to_float(ctx, t1);
        let t1 = q.bound(ctx, t1);
        let t2 = q.int_to_float(ctx, t2);
        let t2 = q.bound(ctx, t2);
        for x in row {
            let xb = q.bound(ctx, *x);
            let below_t1 = q.lt(ctx, xb, t1);
            let below_t2 = q.lt(ctx, xb, t2);
            // t1 <= x < t2: x + 5; x >= t2: 30; otherwise 0.
            let keep = q.gate.mul_not(ctx, below_t1, below_t2);
            let thirty = q.gate.mul_add(ctx, below_t2, Constant(fe(-30 * SCALE)), Constant(fe(30 * SCALE)));
            let plus_five = q.add(ctx, *x, Constant(fe(5 * SCALE)));
            result.push(q.gate.mul_add(ctx, keep, plus_five, thirty));
        }
    }
    out.floats(&result);
    Ok(())
}
