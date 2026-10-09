//! DS-1000 case 0412, halo2-base LITERAL baseline (LITERAL_BASELINES.md).
//! Literal translation of kernel.py: `result = x[x >= 0]` on `Float[10]`
//! (data-dependent length). `x[mask]` has no source to copy (L5): the
//! textbook algorithm is a running count of kept elements, then for each
//! output slot `j` the sum over `i` of `[kept_i and count_i == j] * x_i` (an
//! equality test per pair) -- NOT `q.compact`'s triangular shift network,
//! which L6 forbids. `count_i` is the number of kept elements before
//! position `i` (`prefix[i]`); an element kept at position `i` lands in
//! output slot `prefix[i]`.
//! `x[i] >= 0` is the one and only source comparison on each `x[i]`, so the
//! single domain-check window proof from `load_floats_bounded` is used
//! directly for it (CONVENTIONS.md 1.6: the domain check is not "reuse";
//! using a second, fresh `q.bound()` on the same input cell here would only
//! add a redundant window check the checker counts and rejects).

use crate::prelude::*;

const N: usize = 10;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats_bounded(ctx, &x);

    // kept_i = x_i >= 0, i.e. not (x_i < 0).
    let mut keep = Vec::with_capacity(N);
    for xi in &x {
        let neg = q.is_neg(ctx, *xi);
        keep.push(q.gate.not(ctx, neg));
    }

    // prefix[i] = number of kept elements among x_0 .. x_{i-1}.
    let mut prefix = Vec::with_capacity(N + 1);
    prefix.push(q.int_const(ctx, 0));
    for i in 0..N {
        prefix.push(q.gate.add(ctx, prefix[i], keep[i]));
    }
    let count = prefix[N];

    // out[j] = sum_i [kept_i and prefix_i == j] * x_i.
    let mut outvals = Vec::with_capacity(N);
    for j in 0..N {
        let j_const = q.int_const(ctx, j as i64);
        let mut terms = Vec::with_capacity(N);
        for i in 0..N {
            let is_slot = q.eq(ctx, prefix[i], j_const);
            let kept_here = q.gate.mul(ctx, keep[i], is_slot);
            terms.push(q.gate.mul(ctx, kept_here, x[i].cell));
        }
        outvals.push(q.sum(ctx, terms));
    }

    out.dynamic(count, &outvals, Kind::Float, 1);
    Ok(())
}
