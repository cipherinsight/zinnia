//! DS-1000 case 0398, halo2-base baseline.
//! Kernel: `B[0] = a*A[0]`, `B[k] = a*A[k] + b*B[k-1]` on `A: Float[30]`, `a, b: Integer`.
//! An integer times a Q32 float is exact (no rescale), so each step is one
//! field inner product `a*A[k] + b*B[k-1]`; nothing is witnessed.

use crate::prelude::*;

const N: usize = 30;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let big_a = inputs.floats("A", &[N])?;
    let (a, b) = (inputs.int("a")?, inputs.int("b")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let big_a = q.load_floats(ctx, &big_a);
    let (a, b) = (q.load_int(ctx, a), q.load_int(ctx, b));
    let mut big_b: Vec<Cell> = Vec::with_capacity(N);
    big_b.push(q.mul_int(ctx, big_a[0], a));
    for k in 1..N {
        let step = q.gate.inner_product(ctx, [Existing(a), Existing(b)], [Existing(big_a[k]), Existing(big_b[k - 1])]);
        big_b.push(step);
    }
    out.floats(&big_b);
    Ok(())
}
