// pythran_deuxd_convolution__conv: literal translation of kernel.py (LITERAL_BASELINES.md).
// Clamp indices are compile-time numbers; retain all four source loops and round every product.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const SX0: usize = 16; // = x_0
const SX1: usize = 16; // = x_1
const SW0: usize = 5; // = weights_0
const SW1: usize = 5; // = weights_1

/// The kernel's `clamp(i, offset - SW0 // 2, SW0)`.
fn clamp(i: usize, offset: usize) -> usize {
    (i + offset).saturating_sub(SW0 / 2).min(SW0)
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[SX0, SX1])?;
    let weights = inputs.floats("weights", &[SW0, SW1])?;
    if SW0 >= SX0 || SW0 >= SX1 {
        return Err(Error::new("x[SW0, SW0] is out of bounds: NumPy raises IndexError"));
    }
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // x[r][c] for r, c <= SW0, the part the kernel reads.
    let x: Vec<Vec<Cell>> =
        (0..=SW0).map(|r| (0..=SW0).map(|c| q.load_float(ctx, x[r * SX1 + c])).collect()).collect();
    let weights = q.load_floats(ctx, &weights);
    let mut result = vec![ctx.load_zero(); SX0*SX1];
    for i in 0..SX0 { for j in 0..SX1 { for ii in 0..SW0 { for jj in 0..SW1 {
        let idx = (clamp(i,ii),clamp(j,jj));
        let product = q.mul(ctx,x[idx.0][idx.1],weights[ii*SW1+jj]).cell;
        result[i*SX1+j] = q.add(ctx,result[i*SX1+j],product);
    } } } }
    out.floats(&result); Ok(())
}
