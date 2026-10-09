//! pythran_projection_simplex__projection_simplex (Pythran projection_simplex), halo2-base baseline.
//! Kernel: Euclidean projection of `v: Float[N]` onto the simplex of radius
//! `z` (an integer): `u = sort(v)` descending, `cssv = cumsum(u) - z`,
//! `cond[k] = u[k] - cssv[k] / (k + 1) > 0`, `rho` the last `k + 1` with
//! `cond[k]` (no such `k` raises `IndexError`: no witness),
//! `theta = cssv[rho - 1] / rho`, output `max(v - theta, 0)`.
//! The sort is Batcher's merge-exchange network on the window offsets
//! (`q.sort`), reversed. `cssv[k] / (k + 1)` divides by an integer
//! constant (`q.floor_div_const`; by 1 it is `cssv[0]` itself), and
//! `theta` is the same quotient at `k = rho - 1`, so it is selected from
//! them by the one-hot of the last true condition
//! (`cond[k] * prod_{m > k} (1 - cond[m])`), whose sum must be 1.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = v_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let v = inputs.floats("v", &[N])?;
    let z = inputs.int("z")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let v = q.load_floats_bounded(ctx, &v);
    let z = q.load_int(ctx, z);
    let u: Vec<I64> = q.sort(ctx, &v).into_iter().rev().collect();
    // cssv[k] / (k + 1) and cond[k]
    let z_float = q.int_to_float(ctx, z);
    let mut cumsum: Option<Cell> = None;
    let mut quotients: Vec<Cell> = Vec::with_capacity(N);
    let mut cond: Vec<Cell> = Vec::with_capacity(N);
    for (k, uk) in u.iter().enumerate() {
        let sum = match cumsum {
            None => uk.cell,
            Some(s) => q.add(ctx, s, uk.cell),
        };
        cumsum = Some(sum);
        let cssv = q.sub(ctx, sum, z_float);
        let quotient = if k == 0 { q.bound(ctx, cssv) } else { q.floor_div_const(ctx, cssv, k as u64 + 1).0 };
        let gap = q.sub(ctx, uk.cell, quotient.cell);
        let gap = q.bound(ctx, gap);
        cond.push(q.gt_const(ctx, gap, 0));
        quotients.push(quotient.cell);
    }
    // The one-hot of the last k with cond[k].
    let mut last = vec![cond[N - 1]; N];
    let mut none_after = q.gate.not(ctx, cond[N - 1]);
    for k in (0..N - 1).rev() {
        last[k] = q.gate.mul(ctx, cond[k], none_after);
        if k > 0 {
            none_after = q.gate.sub_mul(ctx, none_after, none_after, cond[k]);
        }
    }
    let hits = q.sum(ctx, last.iter().copied());
    q.gate.assert_is_const(ctx, &hits, &Fr::ONE);
    let theta = q.gate.inner_product(ctx, last.iter().map(|h| Existing(*h)), quotients.iter().map(|x| Existing(*x)));
    // np.maximum(v - theta, 0)
    let zero = q.bounded_const(ctx, 0);
    let w: Vec<Cell> = v
        .iter()
        .map(|vi| {
            let gap = q.sub(ctx, vi.cell, theta);
            let gap = q.bound(ctx, gap);
            q.max(ctx, gap, zero).cell
        })
        .collect();
    out.floats(&w);
    Ok(())
}
