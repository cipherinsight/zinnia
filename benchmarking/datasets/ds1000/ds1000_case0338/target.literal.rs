//! DS-1000 case 0338: literal translation of kernel.py (LITERAL_BASELINES.md).
//! fliplr and diag are views, materialized as wire aliases; domain-check only the selected elements (L7).
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let diag: Vec<f64> = (0..8).map(|i| a[i * 8 + i]).collect();
    // np.fliplr is a view: reverse column wires for every row.
    let flipped: Vec<Vec<f64>> = a.chunks(8).map(|row| row.iter().rev().copied().collect()).collect();
    let anti: Vec<f64> = (0..8).map(|i| flipped[i][i]).collect();
    let rows = [diag, anti];
    let result = q.load_floats(ctx, &rows.concat());
    out.floats(&result);
    Ok(())
}
