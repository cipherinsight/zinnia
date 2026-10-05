//! DS-1000 case 0443, halo2-base baseline.
//! Kernel: `A[np.in1d(A, B)]` on `A: Integer[20]`, `B: Integer[3]` (data-dependent length).
//! Output: the count of kept values, then the kept values in order, zero-padded
//! to 20.
//! Membership: `A_i` is in `B` iff `z(A_i) = 0` for
//! `z(X) = (X - B0)(X - B1)(X - B2) = X^3 - e1 X^2 + e2 X - e3` (a field has no
//! zero divisors, and distinct integers are distinct field elements). The
//! coefficients are computed once; each `A_i` costs a Horner evaluation and
//! one zero test instead of three equality tests.
//! Compaction: `q.compact` (a triangular shift network, no equality tests).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.ints("A", &[20])?, inputs.ints("B", &[3])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let b = q.load_ints(ctx, &b);
    let e1 = q.sum(ctx, b.iter().copied());
    let b01 = q.gate.mul(ctx, b[0], b[1]);
    let b0_plus_b1 = q.add(ctx, b[0], b[1]);
    let e2 = q.gate.mul_add(ctx, b[2], b0_plus_b1, b01);
    let e3 = q.gate.mul(ctx, b01, b[2]);
    let minus_e3 = q.neg(ctx, e3);
    let mut keep = Vec::with_capacity(20);
    let mut weights = Vec::with_capacity(20);
    for ai in &a {
        let z = q.sub(ctx, *ai, e1);
        let z = q.gate.mul_add(ctx, z, *ai, e2);
        let z = q.gate.mul_add(ctx, z, *ai, minus_e3);
        let in_b = q.gate.is_zero(ctx, z);
        weights.push(vec![Some(q.gate.mul(ctx, in_b, *ai))]);
        keep.push(in_b);
    }
    let count = q.sum(ctx, keep.iter().copied());
    let kept = q.compact(ctx, &keep, &weights);
    out.dynamic(count, &kept, Kind::Int, 1);
    Ok(())
}

