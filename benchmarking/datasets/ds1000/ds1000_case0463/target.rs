//! DS-1000 case 0463, halo2-base baseline.
//! Kernel: the (20, 1) column variant of 0462: `selection[1:] = a[1:] !=
//! a[:-1]` (first row selected), `selection &= a != 0`,
//! `a[selection].reshape(-1, 1)` on `Integer[20, 1]` (data-dependent length).
//! Output: the count of kept rows, then the kept values in order,
//! zero-padded to 20 rows of one. Only equalities decide the selection (no
//! range check). Compaction by a running one-hot of the number of kept values
//! before element `i` (one select per pair, no equality tests).

use crate::prelude::*;
use halo2_base::QuantumCell;

const N: usize = 20;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[N, 1])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let mut keep = Vec::with_capacity(N);
    for i in 0..N {
        let zero = q.gate.is_zero(ctx, a[i]);
        let nonzero = q.gate.not(ctx, zero);
        keep.push(if i == 0 {
            nonzero
        } else {
            let repeat = q.eq(ctx, a[i], a[i - 1]);
            q.gate.mul_not(ctx, repeat, nonzero)
        });
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
