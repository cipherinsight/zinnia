//! DS-1000 case 0414, halo2-base LITERAL baseline (LITERAL_BASELINES.md).
//! Literal translation of kernel.py:
//!   bin_size = 4
//!   bin_data_mean = data[:(8 // 4) * 4].reshape(-1, 4).mean(axis=1)
//! `data.size // bin_size`, `* bin_size` and the reshape bound are
//! Python-level constants (shapes; L3 allows compile-time arithmetic on
//! them), so the slice is all of `data` and the reshape is exact: two bins of
//! 4. `mean(axis=1)` is a reduction (L5): a running sum per bin, then one
//! division by the constant `bin_size` -- the shared constant-divisor gadget,
//! which is the arithmetic contract (CONVENTIONS.md section 2), not an
//! optimization.

use crate::prelude::*;

const BIN_SIZE: usize = 4;
const BINS: usize = 2; // 8 / BIN_SIZE

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    let mut means = Vec::with_capacity(BINS);
    for b in 0..BINS {
        let mut total = q.int_const(ctx, 0);
        for k in 0..BIN_SIZE {
            total = q.gate.add(ctx, total, data[b * BIN_SIZE + k]);
        }
        means.push(q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell);
    }
    out.floats(&means);
    Ok(())
}
