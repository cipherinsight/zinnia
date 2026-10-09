//! DS-1000 case 0497: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Pad materializes all output slots; the scalar fill is converted once to the array dtype.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[42, 10])?;
    let e = inputs.int("element")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let element = q.load_int(ctx, e);
    let fill = q.int_to_float(ctx, element);
    // shape = (93, 13); result = np.pad(...)
    let mut result = vec![fill; 93 * 13];
    for i in 0..42 {
        for j in 0..10 {
            result[i * 13 + j] = a[i * 10 + j];
        }
    }
    out.floats(&result);
    Ok(())
}
