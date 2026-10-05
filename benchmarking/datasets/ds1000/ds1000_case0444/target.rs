//! DS-1000 case 0444, halo2-base baseline.
//! Kernel: `A[(A > B[0]) & (A < B[1]) | (A > B[1]) & (A < B[2])]` on
//! `Integer[20]`, `Integer[3]` (data-dependent length).
//! Output: the count of kept values, then the kept values in order,
//! zero-padded to 20. Each value is bounded once; `A > B[1]` is
//! `1 - [A < B[1]] - [A == B[1]]` (an equality instead of a fourth
//! comparison), and the two conjunctions exclude each other (`A < B[1]`
//! against `A > B[1]`), so the `|` is their sum. Compaction by a running
//! one-hot of the number of kept values before element `i` (one select per
//! pair, no equality tests).

use crate::prelude::*;

const N: usize = 20;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a_in, b_in) = (inputs.ints("A", &[N])?, inputs.ints("B", &[3])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a_in);
    let b = q.load_ints(ctx, &b_in);
    let bb = q.bound_all(ctx, &b);
    let mut keep = Vec::with_capacity(N);
    for x in &a {
        let xb = q.bound(ctx, *x);
        let above0 = q.lt(ctx, bb[0], xb);
        let below1 = q.lt(ctx, xb, bb[1]);
        let at1 = q.eq(ctx, *x, b[1]);
        let below2 = q.lt(ctx, xb, bb[2]);
        let above1 = q.gate.not(ctx, below1);
        let above1 = q.sub(ctx, above1, at1);
        let first = q.gate.mul(ctx, above0, below1);
        keep.push(q.gate.mul_add(ctx, above1, below2, first));
    }
    let (count, slots) = q.select_kept(ctx, &a, &keep);
    out.dynamic(count, &slots, Kind::Int, 1);
    Ok(())
}

