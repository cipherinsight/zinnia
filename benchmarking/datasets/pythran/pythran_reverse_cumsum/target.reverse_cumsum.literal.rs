// pythran_reverse_cumsum__reverse_cumsum: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reverse is wire movement; cumsum is a running accumulator in reversed index order.
use crate::prelude::*;
const N: usize = 10; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let reverse: Vec<Cell> = x.iter().rev().copied().collect();
    let mut sum = q.float_const(ctx, 0.0);
    let mut cumulative = Vec::with_capacity(N);
    for v in reverse {
        sum = q.add(ctx, sum, v);
        cumulative.push(sum);
    }
    let result: Vec<Cell> = cumulative.into_iter().rev().collect();
    out.floats(&result);
    Ok(())
}
