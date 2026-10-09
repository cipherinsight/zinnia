// DS-1000 case 0365, literal translation of kernel.py (LITERAL_BASELINES.md).
// np.insert uses its definition: normalize pos in [-100,100], then select prefix / inserted value / shifted suffix for each output.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[100])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let pos = q.load_int(ctx, inputs.int("pos")?);
    let element = q.load_int(ctx, inputs.int("element")?);
    let elements = vec![q.int_to_float(ctx, element)];
    q.bound(ctx, elements[0]); // The inserted float must fit the Q32 window.
    // Normalize negative insertion positions; unlike an indexed read, the endpoint 100 is valid.
    let pw = q.bound(ctx, pos);
    let zero = q.int_const(ctx, 0);
    let zw = q.bound(ctx, zero);
    let neg = q.lt(ctx, pw, zw);
    let wrapped = q.add(ctx, pos, Constant(fe(100)));
    let point = q.gate.select(ctx, wrapped, pos, neg);
    q.range.check_less_than_safe(ctx, point, 101);
    let mut result = Vec::new();
    for k in 0..101 {
        let kcell = q.int_const(ctx, k as i64);
        let kw = q.bound(ctx, kcell);
        let pw = q.bound(ctx, point);
        let before = q.lt(ctx, kw, pw);
        let equal = q.eq(ctx, point, Constant(fe(k as i128)));
        for j in 0..1 {
            let left = if k < 100 { a[k * 1 + j] } else { zero };
            let right = if k > 0 { a[(k - 1) * 1 + j] } else { zero };
            let tail = q.gate.select(ctx, elements[j], right, equal);
            result.push(q.gate.select(ctx, left, tail, before));
        }
    }
    out.floats(&result);
    Ok(())
}
