//! DS-1000 case 0477, optimized halo2-base baseline.
//! np.diag of a length-four vector: only the four diagonal values are public.
//! SVD belongs to DS-1000 input preprocessing, not this kernel.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("i", &[4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let values = q.load_floats(ctx, &raw);
    for r in 0..4 {
        for c in 0..4 {
            if r == c { out.float(values[r]); }
            else { out.const_floats(&[0.0]); }
        }
    }
    Ok(())
}
