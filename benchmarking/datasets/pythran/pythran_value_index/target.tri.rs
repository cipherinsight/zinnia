//! pythran_value_index__tri (Pythran value_index), halo2-base baseline.
//! Kernel: `score[np.argsort(score[:, 1])]` for `score: Int[N, COLS]`, the
//! rows reordered by their second column; `score_t` is written, then
//! rebound, and never read (not loaded).
//! NumPy's tie order is not specified (its default sort is not stable on
//! every platform); like CONVENTIONS section 3, ties keep the row order (a
//! stable argsort).
//! The order is a hint, as a 0/1 matrix `hot` whose rows each hold one 1:
//! output row `k` is `sum_i hot[k][i] * score[i]` (an inner product per
//! output value), and its key `score[i, 1] * 2^INDEX_BITS + i` (through the
//! same row of `hot`) must exceed the previous row's: one `KEY_BITS`-bit
//! check of each step. The keys are distinct and strictly increase, so every
//! input row is picked once, in the stable order: the hint is pinned.
//! Measured against Batcher's merge-exchange network moving the keyed rows
//! (8,165 advice cells at the fixture's shape against 7,543 for this, before
//! dropping its column sums, which the increasing keys make redundant).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = score_0
const COLS: usize = 10; // = score_1
const INDEX_BITS: usize = 4; // = clog2(score_0)
const KEY_BITS: usize = 64 + INDEX_BITS;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let score = inputs.ints("score", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // The hint: the stable ascending order of the second column.
    let mut order: Vec<usize> = (0..N).collect();
    order.sort_by_key(|&i| score[i * COLS + 1]);

    let score = q.load_ints(ctx, &score);
    let keys: Vec<Cell> = (0..N)
        .map(|i| q.gate.mul_add(ctx, score[i * COLS + 1], Constant(fe(1 << INDEX_BITS)), Constant(fe(i as i128))))
        .collect();
    let mut sorted = Vec::with_capacity(N * COLS);
    let mut previous: Option<Cell> = None;
    for &picked in &order {
        // Pinned by the bit and sum checks and the increasing keys (header).
        let hot: Vec<Cell> = (0..N).map(|i| ctx.load_witness(Fr::from((i == picked) as u64))).collect();
        for h in &hot {
            q.gate.assert_bit(ctx, *h);
        }
        let ones = q.sum(ctx, hot.iter().copied());
        q.gate.assert_is_const(ctx, &ones, &Fr::ONE);
        let pick = |ctx: &mut Ctx, column: &mut dyn Iterator<Item = Cell>| {
            q.gate.inner_product(ctx, hot.iter().map(|h| Existing(*h)), column.map(Existing))
        };
        for c in 0..COLS {
            sorted.push(pick(ctx, &mut (0..N).map(|i| score[i * COLS + c])));
        }
        let key = pick(ctx, &mut keys.iter().copied());
        if let Some(previous) = previous {
            let step = q.sub(ctx, key, previous);
            let step = q.sub(ctx, step, Constant(Fr::ONE));
            q.range.range_check(ctx, step, KEY_BITS);
        }
        previous = Some(key);
    }
    out.ints(&sorted);
    Ok(())
}
