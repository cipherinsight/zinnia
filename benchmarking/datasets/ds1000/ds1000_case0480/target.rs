//! DS-1000 case 0480, halo2-base baseline.
//! Kernel: `result = ((x == a) & (y == b)).argmax()` on `x, y: Integer[8]`, then
//! `result = -1` if `x[result] != a or y[result] != b`.
//! This is the first index `i` with `x[i] == a and y[i] == b`, or -1 if there is
//! none: when a match exists, argmax returns the first one and the re-check
//! passes; when none does, argmax returns 0 and the re-check at index 0 fails.
//! So the circuit does not gather `x[result]`, `y[result]`: a select chain from
//! the last index down to the first, starting at -1, gives the same result for
//! every input.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (x, y) = (inputs.ints("x", &[8])?, inputs.ints("y", &[8])?);
    let (a, b) = (inputs.int("a")?, inputs.int("b")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (x, y) = (q.load_ints(ctx, &x), q.load_ints(ctx, &y));
    let (a, b) = (q.load_int(ctx, a), q.load_int(ctx, b));
    let mut result = q.int_const(ctx, -1);
    for i in (0..8).rev() {
        let x_hit = q.eq(ctx, x[i], a);
        let y_hit = q.eq(ctx, y[i], b);
        let hit = q.gate.and(ctx, x_hit, y_hit);
        result = q.gate.select(ctx, Constant(fe(i as i128)), result, hit);
    }
    out.int(result);
    Ok(())
}
