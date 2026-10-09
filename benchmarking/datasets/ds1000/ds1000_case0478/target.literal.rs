//! DS-1000 case 0478: literal translation of kernel.py (LITERAL_BASELINES.md).
//! einsum ii->i is a writable diagonal view: the last assignment writes through to a.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    // einsum('ii->i') reads only the diagonal and returns an aliasing view.
    let result = q.load_ints(ctx, &[raw[0], raw[5], raw[10], raw[15]]);
    let save = result.clone();
    let mut a = vec![ctx.load_zero(); 16]; // a[...] = 0
    for i in 0..4 {
        a[i * 5] = save[i];
    } // result[...] = save
    for i in 0..16 {
        if i % 5 == 0 {
            out.int(a[i]);
        } else {
            out.const_ints(&[0]);
        }
    }
    Ok(())
}
