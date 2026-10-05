//! DS-1000 case 0412, halo2-base baseline.
//! Kernel: `x[x >= 0]` on `Float[10]` (data-dependent length).
//! Output: the count of kept values, then the kept values in order, zero-padded
//! to 10. Kept value `i` goes to slot `p_i` (the number of kept values before
//! it): slot `j` is `sum_i keep_i * [p_i == j] * x_i` over `i >= j`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut keep = Vec::with_capacity(10);
    let mut before = Vec::with_capacity(10); // p_i
    let mut count = q.int_const(ctx, 0);
    for xi in &x {
        let xi = q.bound(ctx, *xi);
        let neg = q.is_neg(ctx, xi);
        let k = q.gate.not(ctx, neg);
        before.push(count);
        count = q.add(ctx, count, k);
        keep.push(k);
    }
    let mut slots = Vec::with_capacity(10);
    for j in 0..10 {
        let mut slot = q.int_const(ctx, 0);
        for i in j..10 {
            let at_j = q.eq(ctx, before[i], Constant(fe(j as i128)));
            let take = q.gate.and(ctx, keep[i], at_j);
            slot = q.gate.mul_add(ctx, take, x[i], slot);
        }
        slots.push(slot);
    }
    out.dynamic(count, &slots, Kind::Float, 1);
    Ok(())
}
