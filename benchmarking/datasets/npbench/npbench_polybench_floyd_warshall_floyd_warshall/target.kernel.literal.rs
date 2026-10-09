//! npbench_polybench_floyd_warshall_floyd_warshall__kernel:
//! literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize the full outer add, then elementwise minimum, then slice assignment.
//! Each minimum bounds both operands afresh. Compared sums must fit signed64.
use crate::prelude::*;
const N: usize = 16; // = path_0
const COLS: usize = 16; // = path_1
const CELLS: usize = N * COLS;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.ints("path", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut path = q.load_ints(ctx, &values);
    for k in 0..N {
        // np.add.outer(path[:, k], path[k, :])
        let mut outer = Vec::with_capacity(CELLS);
        for i in 0..N {
            for j in 0..COLS {
                outer.push(q.add(ctx, path[i * COLS + k], path[k * COLS + j]));
            }
        }
        // np.minimum(path[:], outer)
        let mut minimum = Vec::with_capacity(CELLS);
        for index in 0..CELLS {
            let a = q.bound(ctx, path[index]);
            let b = q.bound(ctx, outer[index]);
            let take_b = q.lt(ctx, b, a);
            minimum.push(q.gate.select(ctx, b.cell, a.cell, take_b));
        }
        // path[:] = minimum
        path = minimum;
    }
    out.ints(&path);
    Ok(())
}
