//! DS-1000 case 0373: optimized baseline.
//! The unused sort is eliminated; input-domain checks retain the source read domain.
//! Constant ratios use floor(i * 2^32 / 50) in both variants, with no public cells.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let grades = inputs.floats("grades", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let grades = q.load_floats(ctx, &grades);
    let result: Vec<f64> = (1..=50).map(|i| ((i * SCALE) / 50) as f64 / SCALE as f64).collect();
    let _ = grades;
    out.const_floats(&result);
    Ok(())
}
