//! DS-1000 case 0326: literal translation of kernel.py (LITERAL_BASELINES.md).
//! No shared arcsin entry exists: local half-angle arcsin(v)=2 atan(v/(1+sqrt(1-v*v))).
//! Use the baseline's degree-9 atan approximation coefficients and exact raw integer root
//! as the arithmetic contract for this library operation, through shared gadgets only.
//! Materialize arcsin in radians, then separately multiply by Q32 180/pi for degrees.
use crate::prelude::*;
const ATAN_COEFS: [f64; 10] = [
    -0.001804437510348226, 0.010962195309438757, -0.0312640164156834,
    0.0580399231381612, -0.08407845425531969, 0.10953436857600397,
    -0.14264386880640922, 0.19998541917604715, -0.33333296696567116,
    0.9999999990510089,
];
fn arcsin(q: &Q32, ctx: &mut Ctx, value: Cell) -> Cell {
    // NumPy arcsin is finite only on [-1,1].
    let shifted = q.add(ctx, value, Constant(fe(SCALE)));
    q.range.check_less_than_safe(ctx, shifted, 2 * SCALE as u64 + 1);
    // Exact raw root is the arithmetic definition within this arcsin gadget,
    // not a fusion across source NumPy calls.
    let square = q.gate.mul(ctx, value, value);
    let radicand = q.sub(ctx, Constant(fe(SCALE * SCALE)), square);
    let root = q.isqrt(ctx, radicand).cell;
    let denominator = q.add(ctx, Constant(fe(SCALE)), root);
    let denominator_b = q.bound(ctx, denominator);
    let t = q.div_pos(ctx, value, denominator_b).cell;
    let t2 = q.mul(ctx, t, t).cell;
    let polynomial = q.polynomial(ctx, t2, &ATAN_COEFS);
    let angle = q.mul(ctx, t, polynomial).cell;
    q.mul_int(ctx, angle, Constant(fe(2)))
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.float("value")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let value = q.load_float(ctx, input);
    let radians = arcsin(&q, ctx, value);
    let result = q.mul(ctx, radians, Constant(fe(quantize(180.0 / std::f64::consts::PI)))).cell;
    out.float(result);
    Ok(())
}
