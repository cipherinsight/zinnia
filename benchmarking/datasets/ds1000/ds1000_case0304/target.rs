//! DS-1000 case 0304, halo2-base baseline.
//! Kernel: `np.reshape(A[23 - 20:][::-1], (-1, 5))` on `Float[23]` (`ncol = 5`
//! is a constant, so 20 cells are kept): `A[22], A[21], ..., A[3]` as a (4, 5)
//! array. The outputs are input cells, so the circuit loads only those 20 and
//! exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[23])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let reversed: Vec<f64> = a[3..].iter().rev().copied().collect();
    let reversed = q.load_floats(ctx, &reversed);
    out.floats(&reversed);
    Ok(())
}
