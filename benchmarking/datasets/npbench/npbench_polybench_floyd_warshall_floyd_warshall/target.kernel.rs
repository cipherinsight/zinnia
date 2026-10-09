//! Floyd-Warshall: snapshot each vectorized outer-add/minimum update.
//! Exact integer sums must fit signed64 when compared; no NumPy wraparound.
//! Reuse window proofs across iterations; no assumptions about diagonal/signs.
use crate::prelude::*;
const N: usize = 16; // = path_0
const COLS: usize = 16; // = path_1
const CELLS: usize = N * COLS;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let values = inputs.ints("path", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut path = q.load_ints_bounded(ctx, &values);
    for k in 0..N {
        let mut next = Vec::with_capacity(CELLS);
        for i in 0..N {
            for j in 0..COLS {
                let sum = q.add(ctx, path[i * COLS + k].cell, path[k * COLS + j].cell);
                let sum = q.bound(ctx, sum);
                next.push(q.min(ctx, path[i * COLS + j], sum));
            }
        }
        path = next;
    }
    out.ints(&path.iter().map(|v| v.cell).collect::<Vec<_>>());
    Ok(())
}
