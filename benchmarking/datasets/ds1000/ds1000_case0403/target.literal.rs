//! DS-1000 case 0403: literal translation of kernel.py (LITERAL_BASELINES.md).
//! ravel_multi_index is the C-stride defining sum in index order.
//! The standard inner-product gadget loads the five private input operands;
//! these are input plumbing, not computed hints. Dimension checks copied under
//! L7 imply their input domain (CONVENTIONS.md 1.6); a is never read.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let index = inputs.ints("index", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let dims = [8u64, 9, 8, 8, 8];
    let strides: Vec<u64> = (0..5).map(|k| dims[k + 1..].iter().product()).collect();
    let (result, operands) = q.gate.inner_product_left(
        ctx,
        index.iter().map(|v| Witness(fe(*v as i128))),
        strides.iter().map(|v| Constant(Fr::from(*v))),
    );
    for k in 0..5 { assert_below(&q, ctx, operands[k], dims[k]); }
    out.int(result);
    Ok(())
}

// Same input-domain/dimension checks as the optimized baseline (L7).
fn assert_below(q: &Q32, ctx: &mut Ctx, x: Cell, d: u64) {
    let bits = q.range.lookup_bits();
    let table = 1u64 << bits;
    if d > table {
        q.range.check_less_than_safe(ctx, x, d);
        return;
    }
    q.range.range_check(ctx, x, bits);
    if d < table {
        let shifted = q.gate.add(ctx, x, Constant(Fr::from(table - d)));
        q.range.range_check(ctx, shifted, bits);
    }
}
