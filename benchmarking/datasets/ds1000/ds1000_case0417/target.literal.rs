//! DS-1000 case 0417: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Reverse all eight inputs, reshape into two bins of four, and compute each mean.
//! Reversal/slice/reshape are wiring; means sum in index order then floor-divide by four.
use crate::prelude::*;

const BIN_SIZE: usize = 4;
const BINS: usize = 2; // 8 / BIN_SIZE

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    // new_data = data[::-1]; reversal and reshape are wiring.
    let new_data: Vec<Cell> = data.iter().rev().copied().collect();
    let mut means = Vec::with_capacity(BINS);
    for b in 0..BINS {
        let mut total = q.int_const(ctx, 0);
        for k in 0..BIN_SIZE {
            total = q.gate.add(ctx, total, new_data[b * BIN_SIZE + k]);
        }
        means.push(q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell);
    }
    out.floats(&means);
    Ok(())
}
