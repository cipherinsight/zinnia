//! DS-1000 case 0412, halo2-base LITERAL baseline (LITERAL_BASELINES.md).
//! Literal translation of kernel.py: `result = x[x >= 0]` on `Float[10]`
//! (data-dependent length). `x[mask]` has no source to copy (L5): the
//! textbook algorithm is a running count of kept elements, then for each
//! output slot `j` the sum over `i` of `[kept_i and count_i == j] * x_i` (an
//! equality test per pair) -- NOT `q.compact`'s triangular shift network,
//! which L6 forbids. `count_i` is the number of kept elements before
//! position `i` (`prefix[i]`); an element kept at position `i` lands in
//! output slot `prefix[i]`.
//! Each comparison obtains fresh operand bounds as required by L5.

use crate::prelude::*;

const N: usize = 10;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);

    // kept_i = x_i >= 0, i.e. not (x_i < 0).
    let mut keep = Vec::with_capacity(N);
    for xi in &x {
        let xb = q.bound(ctx, *xi);
        let zero = q.int_const(ctx, 0);
        let zb = q.bound(ctx, zero);
        keep.push(q.ge(ctx, xb, zb));
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
            terms.push(q.gate.mul(ctx, kept_here, x[i]));
        }
        outvals.push(q.sum(ctx, terms));
    }

    out.dynamic(count, &outvals, Kind::Float, 1);
    Ok(())
}
