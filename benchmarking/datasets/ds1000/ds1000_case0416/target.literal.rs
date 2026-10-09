//! DS-1000 case 0416: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Slice and reshape are wiring; each mean is a running sum then constant division.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let data = q.load_floats(ctx, &data);
    let mut result = Vec::new();
    for row in 0..8 {
        for b in 0..3 {
            let mut sum = q.int_const(ctx, 0);
            for k in 0..2 { sum = q.add(ctx, sum, data[row * 6 + b * 2 + k]); }
            result.push(q.floor_div_const(ctx, sum, 2).0.cell);
        }
    }
    out.floats(&result);
    Ok(())
}
