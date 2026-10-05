//! DS-1000 case 0441, halo2-base baseline.
//! Kernel: `number in a` on `a: Integer[4]`, `number: Integer`.
//! `number in a` is `(a == number).any()`: the product of the differences
//! `number - a_k` is zero exactly when one of them is (a field has no zero
//! divisors, and distinct integers are distinct field elements), so one zero
//! test replaces four equality tests and their `or`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, number) = (inputs.ints("a", &[4])?, inputs.int("number")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let number = q.load_int(ctx, number);
    let mut product = q.sub(ctx, number, a[0]);
    for ak in &a[1..] {
        let diff = q.sub(ctx, number, *ak);
        product = q.gate.mul(ctx, product, diff);
    }
    let found = q.gate.is_zero(ctx, product);
    out.bool(found);
    Ok(())
}
