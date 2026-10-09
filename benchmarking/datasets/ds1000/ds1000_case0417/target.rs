//! DS-1000 case 0417: optimized baseline.
//! Reverse all eight inputs, reshape into two bins of four, and compute each mean.
//! Reversal/slice/reshape are wiring; means sum in index order then floor-divide by four.
use crate::prelude::*;

const BIN_SIZE: usize = 4;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    // new_data = data[::-1]; reversal and reshape are wiring.
    let new_data: Vec<Cell> = data.iter().rev().copied().collect();
    let means: Vec<Cell> = new_data
        .chunks(BIN_SIZE)
        .map(|bin| {
            let total = q.sum(ctx, bin.iter().copied());
            q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell
        })
        .collect();
    out.floats(&means);
    Ok(())
}
