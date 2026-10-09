//! DS-1000 case 0418: optimized baseline.
//! Reverse the six columns, reshape into pairs, and compute means in row order.
//! Slice/reversal/reshape are wiring; each mean sums then floor-divides by two.
use crate::prelude::*;

const BIN_SIZE: usize = 2;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    // new_data = data[:, ::-1]
    let mut new_data = Vec::with_capacity(48);
    for row in 0..8 {
        for col in 0..6 { new_data.push(data[row * 6 + 5 - col]); }
    }
    // A row's 6 values split into 3 whole bins, so the bins are consecutive
    // pairs of the row-major data.
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
