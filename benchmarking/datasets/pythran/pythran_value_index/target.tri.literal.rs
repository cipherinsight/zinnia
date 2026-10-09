//! pythran_value_index__tri: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Stable adjacent bubble sort: N*(N-1)/2 comparisons carrying indices, then q.index per read.
//! score_t's initial constant-index write is retained as a wire assignment; its input is never read.
use crate::prelude::*;
const N: usize = 10; // = score_0
const COLS: usize = 10; // = score_1
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("score", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let score = q.load_ints(ctx, &raw);
    let _score_t_00 = score[0];
    let mut keys: Vec<Cell> = (0..N).map(|i| score[i * COLS + 1]).collect();
    let mut order: Vec<Cell> = (0..N).map(|i| q.int_const(ctx, i as i64)).collect();
    for pass in 0..N - 1 {
        for j in 0..N - 1 - pass {
            let left = q.bound(ctx, keys[j]);
            let right = q.bound(ctx, keys[j + 1]);
            let swap = q.gt(ctx, left, right);
            let (a, b) = (keys[j], keys[j + 1]);
            keys[j] = q.gate.select(ctx, b, a, swap);
            keys[j + 1] = q.gate.select(ctx, a, b, swap);
            let (a, b) = (order[j], order[j + 1]);
            order[j] = q.gate.select(ctx, b, a, swap);
            order[j + 1] = q.gate.select(ctx, a, b, swap);
        }
    }
    let mut score_t = Vec::new();
    for i in 0..N {
        for j in 0..COLS {
            let column: Vec<Cell> = (0..N).map(|k| score[k * COLS + j]).collect();
            score_t.push(q.index(ctx, &column, order[i]));
        }
    }
    out.ints(&score_t);
    Ok(())
}
