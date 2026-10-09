// DS-1000 case 0402, literal translation of kernel.py (LITERAL_BASELINES.md).
// Fortran ravel is an index-order sum of coordinate times constant stride; coordinate bounds reject NumPy errors. Parameter a is unread.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("index", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let index = q.load_ints(ctx, &raw);
    let dims = [8u64, 9, 8, 8, 8];
    let mut stride = 1;
    let mut result = q.int_const(ctx, 0);
    for k in 0..5 {
        q.range.check_less_than_safe(ctx, index[k], dims[k]);
        let term = q.gate.mul(ctx, index[k], Constant(fe(stride)));
        result = q.add(ctx, result, term);
        stride *= dims[k] as i128;
    }
    out.int(result);
    Ok(())
}
