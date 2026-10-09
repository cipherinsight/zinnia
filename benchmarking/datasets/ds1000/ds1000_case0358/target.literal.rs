// DS-1000 case 0358, literal translation of kernel.py (LITERAL_BASELINES.md).
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
    // a.sum(axis=(1,2)), running sums in index order, then argsort.
    let mut sums = Vec::new();
    for i in 0..3 {
        let mut sum = a[i * 9];
        for j in 1..9 {
            sum = q.add(ctx, sum, a[i * 9 + j]);
        }
        sums.push(sum);
    }
    let (_, index) = bubble(&q, ctx, &sums);
    let mut result = Vec::new();
    for i in 0..3 {
        for j in 0..9 {
            let column: Vec<Cell> = (0..3).map(|k| b[k * 9 + j]).collect();
            result.push(q.index(ctx, &column, index[i]));
        }
    }
    out.ints(&result);
    Ok(())
}
