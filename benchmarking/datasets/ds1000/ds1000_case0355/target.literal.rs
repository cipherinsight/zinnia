// DS-1000 case 0355, literal translation of kernel.py (LITERAL_BASELINES.md).
// Stable adjacent bubble sort uses n(n-1)/2 compare-and-swaps, with fresh bounds per comparison.
use crate::prelude::*;
fn bubble(q: &Q32, ctx: &mut Ctx, values: &[Cell]) -> (Vec<Cell>, Vec<Cell>) {
    let mut keys = values.to_vec();
    let mut indices: Vec<Cell> = (0..keys.len())
        .map(|i| q.int_const(ctx, i as i64))
        .collect();
    for pass in 0..keys.len() - 1 {
        for j in 0..keys.len() - 1 - pass {
            let a = q.bound(ctx, keys[j]);
            let b = q.bound(ctx, keys[j + 1]);
            let swap = q.gt(ctx, a, b);
            let (left, right) = (keys[j], keys[j + 1]);
            keys[j] = q.gate.select(ctx, right, left, swap);
            keys[j + 1] = q.gate.select(ctx, left, right, swap);
            let (left, right) = (indices[j], indices[j + 1]);
            indices[j] = q.gate.select(ctx, right, left, swap);
            indices[j + 1] = q.gate.select(ctx, left, right, swap);
        }
    }
    (keys, indices)
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.floats("a", &[3, 3, 3])?;
    let bv = inputs.ints("b", &[3, 3, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &av);
    let b = q.load_ints(ctx, &bv);
    // argsort(a, axis=0), all nine axis slices.
    let mut sort_indices = vec![q.int_const(ctx, 0); 27];
    for j in 0..3 {
        for k in 0..3 {
            let column: Vec<Cell> = (0..3).map(|i| a[i * 9 + j * 3 + k]).collect();
            let (_, order) = bubble(&q, ctx, &column);
            for i in 0..3 {
                sort_indices[i * 9 + j * 3 + k] = order[i];
            }
        }
    }
    // static_indices = np.indices(a.shape)
    let static_indices: Vec<(usize, usize, usize)> =
        (0..27).map(|p| (p / 9, p / 3 % 3, p % 3)).collect();
    let mut result = Vec::new();
    for p in 0..27 {
        let (_, j, k) = static_indices[p];
        let column: Vec<Cell> = (0..3).map(|i| b[i * 9 + j * 3 + k]).collect();
        result.push(q.index(ctx, &column, sort_indices[p]));
    }
    out.ints(&result);
    Ok(())
}
