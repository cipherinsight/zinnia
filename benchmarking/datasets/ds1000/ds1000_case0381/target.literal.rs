// DS-1000 case 0381, literal translation of kernel.py (LITERAL_BASELINES.md).
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
    let raw = inputs.floats("a", &[100])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let (_, sorted_indices) = bubble(&q, ctx, &a);
    let reversed: Vec<Cell> = sorted_indices.into_iter().rev().collect();
    let result = reversed[..100].to_vec();
    out.ints(&result);
    Ok(())
}
