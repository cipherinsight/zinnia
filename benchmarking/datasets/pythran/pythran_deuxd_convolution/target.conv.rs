//! pythran_deuxd_convolution__conv (Pythran deuxd_convolution), halo2-base baseline.
//! Kernel: on x: Float[SX0, SX1], weights: Float[SW0, SW1],
//! `result[i, j] = sum_{ii, jj} x[r(i, ii), c(j, jj)] * weights[ii, jj]` with
//! the kernel's clamp `r(i, ii) = min(max(0, i + ii - SW0 // 2), SW0)` and the
//! same for columns, also clamped to `SW0` (the kernel passes `sw[0]` for
//! both). The indices are loop constants. Each output accumulates its
//! products into one variable from 0, rounding each scalar product to Q32
//! before adding it, as in the source loop.
//! Only `x[:SW0 + 1, :SW0 + 1]` is read, and only it is loaded. From row
//! `SW0 + SW0 // 2` on, every row index clamps to `SW0`, so those rows of
//! `result` repeat that row's computation, and the same holds for columns:
//! each distinct computation (`ROW_KINDS * COL_KINDS` of them) is done once
//! and its cell output wherever it occurs.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const SX0: usize = 16; // = x_0
const SX1: usize = 16; // = x_1
const SW0: usize = 5; // = weights_0
const SW1: usize = 5; // = weights_1
/// Rows and columns of `result` with distinct index patterns: the clamp
/// saturates from `SW0 + SW0 // 2` on.
const ROW_KINDS: usize = 8; // = min(x_0, weights_0 + weights_0 // 2 + 1)
const COL_KINDS: usize = 8; // = min(x_1, weights_0 + weights_0 // 2 + 1)

/// The kernel's `clamp(i, offset - SW0 // 2, SW0)`.
fn clamp(i: usize, offset: usize) -> usize {
    (i + offset).saturating_sub(SW0 / 2).min(SW0)
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[SX0, SX1])?;
    let weights = inputs.floats("weights", &[SW0, SW1])?;
    if SW0 >= SX0 || SW0 >= SX1 {
        return Err(Error::new("x[SW0, SW0] is out of bounds: NumPy raises IndexError"));
    }
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // x[r][c] for r, c <= SW0, the part the kernel reads.
    let x: Vec<Vec<Cell>> =
        (0..=SW0).map(|r| (0..=SW0).map(|c| q.load_float(ctx, x[r * SX1 + c])).collect()).collect();
    let weights = q.load_floats(ctx, &weights);
    let mut kinds = Vec::with_capacity(ROW_KINDS * COL_KINDS);
    // Exact local CSE: clamping repeats identical x/weight products across
    // outputs. Reuse the rounded product without reassociation or fusion.
    let mut products = std::collections::HashMap::new();
    for i in 0..ROW_KINDS {
        for j in 0..COL_KINDS {
            let mut acc = q.int_const(ctx, 0);
            for ii in 0..SW0 {
                for jj in 0..SW1 {
                    let key = (clamp(i, ii), clamp(j, jj), ii * SW1 + jj);
                    let product = *products.entry(key).or_insert_with(||
                        q.mul(ctx, x[key.0][key.1], weights[key.2]).cell);
                    acc = q.add(ctx, acc, product);
                }
            }
            kinds.push(acc);
        }
    }
    for i in 0..SX0 {
        let row: Vec<Cell> = (0..SX1).map(|j| kinds[i.min(ROW_KINDS - 1) * COL_KINDS + j.min(COL_KINDS - 1)]).collect();
        out.floats(&row);
    }
    Ok(())
}
