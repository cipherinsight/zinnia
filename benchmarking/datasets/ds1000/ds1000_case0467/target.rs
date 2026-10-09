//! DS-1000 case 0467: optimized baseline.
//! Clipped 3x3 windows on a 4x4 integer input, row-first window order.
//! Static slices alias input wires; only the 16 input-domain checks need constraints.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &a);
    let mut result = Vec::with_capacity(100);
    for i in 0usize..4 {
        for j in 0usize..4 {
            for r in i.saturating_sub(1)..4.min(i + 2) {
                for c in j.saturating_sub(1)..4.min(j + 2) {
                    result.push(a[4 * r + c]);
                }
            }
        }
    }
    out.ints(&result);
    Ok(())
}
