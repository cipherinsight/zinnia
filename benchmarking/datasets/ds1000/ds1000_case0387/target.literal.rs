// DS-1000 case 0387, literal translation of kernel.py (LITERAL_BASELINES.md).
// Each reshape and swapaxes has its full intermediate wire array; constant permutations need no arithmetic.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[100, 200])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let reshaped = a; // reshape preserves row-major storage
    let mut swapped = Vec::with_capacity(20000);
    for i in 0..50 {
        for j in 0..100 {
            for r in 0..2 {
                for s in 0..2 {
                    swapped.push(reshaped[((i * 2 + r) * 100 + j) * 2 + s]);
                }
            }
        }
    }
    let result = swapped; // final reshape preserves storage
    out.floats(&result);
    Ok(())
}
