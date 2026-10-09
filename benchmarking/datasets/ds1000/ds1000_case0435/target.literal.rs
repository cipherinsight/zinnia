//! DS-1000 case 0435: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Constant writes are wiring; overwritten inputs are unread and constant output segments are omitted per L7.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.int_const(ctx, 0);
    let mut a = vec![zero; 16];
    // Overwritten input cells are never read; load only surviving cells.
    for row in [0, 2, 3] {
        for col in 1..4 { a[row * 4 + col] = q.load_int(ctx, av[row * 4 + col]); }
    }
    for col in 0..4 { a[4 + col] = zero; }
    for row in 0..4 { a[row * 4] = zero; }
    out.const_ints(&[0]);
    out.ints(&a[1..4]);
    out.const_ints(&[0; 5]);
    out.ints(&a[9..12]);
    out.const_ints(&[0]);
    out.ints(&a[13..16]);
    Ok(())
}
