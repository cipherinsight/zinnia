// DS-1000 case 0389, literal translation of kernel.py (LITERAL_BASELINES.md).
// Each reshape, swapaxes and transpose has its full intermediate wire array; constant permutations need no arithmetic.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[6, 3, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let reshaped = a; // reshape preserves row-major storage
    let mut swapped = Vec::with_capacity(72);
    for i in 0..2 {
        for r in 0..3 {
            for j in 0..3 {
                for s in 0..4 {
                    swapped.push(reshaped[((i * 3 + j) * 3 + r) * 4 + s]);
                }
            }
        }
    }
    let result = swapped;
    out.floats(&result);
    Ok(())
}
