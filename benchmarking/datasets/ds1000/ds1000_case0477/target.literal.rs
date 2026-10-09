//! DS-1000 case 0477, literal translation of kernel.py (LITERAL_BASELINES.md).
//! np.diag constructs a zero matrix and fills its diagonal with input wires.
//! The whole matrix is formed; constant cells are omitted from instances (L7).
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("i", &[4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let values = q.load_floats(ctx, &raw);
    // i = np.diag(i)
    let zero = q.float_const(ctx, 0.0);
    let mut i = vec![zero; 16];
    for k in 0..4 { i[k * 4 + k] = values[k]; }
    for r in 0..4 {
        for c in 0..4 {
            if r == c { out.float(i[r * 4 + c]); }
            else { out.const_floats(&[0.0]); }
        }
    }
    Ok(())
}
