//! DS-1000 case 0418, halo2-base baseline.
//! Kernel: `data[:, ::-1][:, :(6 // 2) * 2].reshape(8, -1, 2).mean(axis=-1)`
//! with the constant `bin_size = 2` on `Float[8, 6]`: row `i`, bin `k` is the
//! mean of `data[i, 5 - 2k]` and `data[i, 4 - 2k]`.
//! A Q32 mean of 2 values is `floor(sum / 2)` (the quotient
//! `floor(sum * 2^32 / (2 * 2^32))`), a division by the constant 2: a 1-bit
//! remainder check and a window bound on the quotient. The reversal is a
//! fixed relabelling of the cells.

use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;
const BIN_SIZE: usize = 2;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[ROWS, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    let mut means = Vec::with_capacity(ROWS * COLS / BIN_SIZE);
    for row in data.chunks(COLS) {
        let reversed: Vec<Cell> = row.iter().rev().copied().collect();
        for bin in reversed.chunks(BIN_SIZE) {
            let total = q.sum(ctx, bin.iter().copied());
            means.push(q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell);
        }
    }
    out.floats(&means);
    Ok(())
}
