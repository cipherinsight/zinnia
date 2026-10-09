//! DS-1000 case 0441: literal translation of kernel.py (LITERAL_BASELINES.md).
//! NumPy array membership is elementwise equality followed by an index-order any reduction.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("a", &[4])?;
    let nv = inputs.int("number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &av);
    let number = q.load_int(ctx, nv);
    let equal: Vec<Cell> = a.iter().map(|x| q.eq(ctx, number, *x)).collect();
    let mut contained = q.int_const(ctx, 0);
    for bit in equal { contained = q.gate.or(ctx, contained, bit); }
    out.bool(contained);
    Ok(())
}
