//! DS-1000 case 0419, halo2-base baseline.
//! Kernel: `data[:, ::-1][:, :(6 // 2) * 2].reshape(8, -1, 2).mean(axis=-1)[:, ::-1]`
//! with the constant `bin_size = 2` on `Float[8, 6]`. Reversing the columns,
//! binning pairs and reversing the bins gives back the pairs in order: row
//! `i`, bin `k` is the mean of `data[i, 2k + 1]` and `data[i, 2k]` (the same
//! function as case 0416).
//! A Q32 mean of 2 values is `floor(sum / 2)` (the quotient
//! `floor(sum * 2^32 / (2 * 2^32))`), a division by the constant 2: a 1-bit
//! remainder check and a window bound on the quotient.

use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const BIN_SIZE: usize = 2;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[ROWS, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    // A row's 6 values split into 3 whole bins, so the bins are consecutive
    // pairs of the row-major data.
    let means: Vec<Cell> = data
        .chunks(BIN_SIZE)
        .map(|bin| {
            let total = q.sum(ctx, bin.iter().rev().copied());
            q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell
        })
        .collect();
    out.floats(&means);
    Ok(())
}
