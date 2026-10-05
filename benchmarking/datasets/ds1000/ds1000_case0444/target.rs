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
use halo2_base::QuantumCell;

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
    let (count, slots) = compact(&q, ctx, &a, &keep);
    out.dynamic(count, &slots, Kind::Int, 1);
    Ok(())
}

/// `xs[keep]` at the front of `xs.len()` slots, zero-padded, and its length.
/// `at[j]` is the one-hot of the number of kept values before element `i`
/// (`at = e_0` for `i = 0`; it moves up one place after a kept element), and
/// element `i` adds `keep_i * x_i` to the slot `at` points at.
fn compact(q: &Q32, ctx: &mut Ctx, xs: &[Cell], keep: &[Cell]) -> (Cell, Vec<Cell>) {
    let n = xs.len();
    let mut terms: Vec<(Vec<QuantumCell<Fr>>, Vec<QuantumCell<Fr>>)> = vec![(vec![], vec![]); n];
    let mut at: Vec<QuantumCell<Fr>> = vec![Constant(Fr::ONE)];
    for i in 0..n {
        let y = q.gate.mul(ctx, keep[i], xs[i]);
        for (j, hit) in at.iter().enumerate() {
            terms[j].0.push(*hit);
            terms[j].1.push(Existing(y));
        }
        if i + 1 < n {
            let k = keep[i];
            let mut next = Vec::with_capacity(at.len() + 1);
            next.push(Existing(q.gate.mul_not(ctx, k, at[0])));
            for j in 1..at.len() {
                next.push(Existing(q.gate.select(ctx, at[j - 1], at[j], k)));
            }
            next.push(Existing(q.gate.mul(ctx, k, *at.last().expect("non-empty"))));
            at = next;
        }
    }
    let slots = terms.into_iter().map(|(hits, ys)| q.gate.inner_product(ctx, hits, ys)).collect();
    (q.sum(ctx, keep.iter().copied()), slots)
}
