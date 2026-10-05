//! DS-1000 case 0399, halo2-base baseline.
//! Kernel: `B[0] = a*A[0]`, `B[1] = a*A[1] + b*B[0]`,
//! `B[k] = a*A[k] + b*B[k-1] + c*B[k-2]` on `A: Float[30]`, `a, b, c: Integer`.
//! An integer times a Q32 float is exact (no rescale), so each step is one
//! field inner product; nothing is witnessed.

use crate::prelude::*;

const N: usize = 30;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let big_a = inputs.floats("A", &[N])?;
    let (a, b, c) = (inputs.int("a")?, inputs.int("b")?, inputs.int("c")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let big_a = q.load_floats(ctx, &big_a);
    let (a, b, c) = (q.load_int(ctx, a), q.load_int(ctx, b), q.load_int(ctx, c));
    let mut big_b: Vec<Cell> = Vec::with_capacity(N);
    big_b.push(q.mul_int(ctx, big_a[0], a));
    big_b.push(q.gate.inner_product(ctx, [Existing(a), Existing(b)], [Existing(big_a[1]), Existing(big_b[0])]));
    for k in 2..N {
        let step = q.gate.inner_product(
            ctx,
            [Existing(a), Existing(b), Existing(c)],
            [Existing(big_a[k]), Existing(big_b[k - 1]), Existing(big_b[k - 2])],
        );
        big_b.push(step);
    }
    out.floats(&big_b);
    Ok(())
}
