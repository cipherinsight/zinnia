//! npbench_deep_learning_softmax_softmax__softmax (NPBench deep_learning/softmax),
//! halo2-base literal baseline. LITERAL_BASELINES.md: direct translation of
//! kernel.py, no optimization.
//!
//! Kernel (four statements, each over the last axis of x: Float[N, H, SN, SM]):
//!   tmp_max = np.max(x, axis=-1, keepdims=True)
//!   tmp_out = np.exp(x - tmp_max)
//!   tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
//!   return tmp_out / tmp_sum
//!
//! Each statement is one full pass over all rows before the next statement
//! starts, computing and keeping the whole intermediate array (L1, L2) rather
//! than the optimized baseline's per-row fusion of all four steps.
//!
//! Translation choices:
//! - `np.max` (L5 reduction row): a running accumulator in index order, one
//!   pairwise `q.max` per remaining element, not the hinted `q.max_of`.
//! - `x - tmp_max`: a plain signed subtraction per element (`q.sub`), then a
//!   fresh `q.bound` (not reusing `tmp_max`'s own window proof from the `np.max`
//!   statement, nor anything from `np.max`'s accumulator) because `canonical_exp`
//!   needs the gap's own window proof.
//! - `np.exp`: the optimized baseline uses the frozen compiler arithmetic family
//!   (`q.canonical_exp_neg`) for a strict arithmetic comparison with Zinnia; the
//!   literal baseline keeps that family for the same arithmetic contract
//!   (LITERAL_BASELINES.md section "fair") but calls the *general* entry point
//!   `q.canonical_exp` on the literal signed gap `x - tmp_max`, not the
//!   gap-is-already-nonnegative-specialised `canonical_exp_neg`, which is paired
//!   with the forbidden `q.max_of` hint and is not what the source computes
//!   (the source never negates anything: `np.exp(x - tmp_max)`, not
//!   `np.exp(-(tmp_max - x))`).
//! - `np.sum`: a running accumulator in index order (`q.sum`, exact for
//!   already-Q32 values, no further rescale: this is an addition, not a
//!   reduction of products).
//! - `tmp_out / tmp_sum`: a plain float division per element. `tmp_sum`'s row
//!   total is freshly bounded (one `q.bound` per row, reused only across the
//!   SM elements of that one broadcast division, i.e. within this one source
//!   statement, exactly as a clipped array's bound is reused across its own
//!   elementwise statement). `q.div_pos` is used, not `q.div_by_abs`: the row
//!   total is a sum of `np.exp` outputs, each in `[0, 2^33)`, and the row's own
//!   maximum contributes exactly `exp(0) = 2^32`, so the total is always
//!   strictly positive by construction of this computation, for every input
//!   (not a fixture-specific fact) -- the same kind of "divisor provably
//!   positive from how it was built" reasoning already accepted for a
//!   `np.sqrt` divisor (`cholesky`), not a reuse of another operation's proof:
//!   `div_pos`'s own remainder check `0 <= r < b` is what rules out `b <= 0`,
//!   with no externally supplied magnitude.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = x_0
const H: usize = 2; // = x_1
const SN: usize = 16; // = x_2
const SM: usize = 16; // = x_3

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N, H, SN, SM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats_bounded(ctx, &x);

    // tmp_max = np.max(x, axis=-1, keepdims=True)
    let tmp_max: Vec<I64> = x
        .chunks(SM)
        .map(|row| {
            let mut largest = row[0];
            for &v in &row[1..] {
                largest = q.max(ctx, largest, v);
            }
            largest
        })
        .collect();

    // tmp_out = np.exp(x - tmp_max)
    let tmp_out: Vec<I64> = x
        .chunks(SM)
        .zip(&tmp_max)
        .flat_map(|(row, largest)| {
            row.iter()
                .map(|v| {
                    let gap = q.sub(ctx, v.cell, largest.cell);
                    let gap = q.bound(ctx, gap);
                    q.canonical_exp(ctx, gap)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    let tmp_sum: Vec<Cell> = tmp_out.chunks(SM).map(|row| q.sum(ctx, row.iter().map(|v| v.cell))).collect();

    // return tmp_out / tmp_sum
    let result: Vec<Cell> = tmp_out
        .chunks(SM)
        .zip(&tmp_sum)
        .flat_map(|(row, total)| {
            let total = q.bound(ctx, *total);
            row.iter().map(|e| q.div_pos(ctx, e.cell, total).cell).collect::<Vec<_>>()
        })
        .collect();
    out.floats(&result);
    Ok(())
}
