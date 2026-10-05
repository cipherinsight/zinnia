//! DS-1000 case 0322, halo2-base baseline.
//! Kernel: `np.argwhere(a == np.min(a))` on `Integer[2, 2]` (data-dependent length).
//! Output: the number of minima, then their `(row, col)` pairs in row-major
//! order, zero-padded to 4 pairs.
//! The minimum is minus the maximum of the negated values, `q.max_of` (a hint
//! pinned by four range checks and a product); `a == min` is `-a == -min`.
//! Compaction: `q.compact` (a triangular shift network, no equality tests);
//! the payload of flat index `k` is the constant pair `(k // 2, k % 2)`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[2, 2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let negated: Vec<Cell> = a.iter().map(|x| q.neg(ctx, *x)).collect();
    let minus_min = q.max_of(ctx, &negated);
    let hit: Vec<Cell> = negated.iter().map(|x| q.eq(ctx, *x, minus_min)).collect();
    let count = q.sum(ctx, hit.iter().copied());
    // (row, col) is 0 or 1, so the weight hit_k * coordinate is hit_k or nothing.
    let weights: Vec<Vec<Option<Cell>>> =
        (0..4).map(|k| vec![(k / 2 == 1).then_some(hit[k]), (k % 2 == 1).then_some(hit[k])]).collect();
    let coords = q.compact(ctx, &hit, &weights);
    out.dynamic(count, &coords, Kind::Int, 2);
    Ok(())
}

