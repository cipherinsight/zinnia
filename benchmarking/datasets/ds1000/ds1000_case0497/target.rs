//! DS-1000 case 0497, halo2-base baseline.
//! Kernel: `np.pad(a, ((0, 93 - 42), (0, 13 - 10)), 'constant',
//! constant_values=element)` on `Float[42, 10]` and an `Integer` `element`
//! (`shape = (93, 13)` is a constant): each row of `a` followed by 3 copies of
//! `float(element)`, then 51 rows of it. The padding depends on the private
//! `element`, so those cells are public copies of the one converted cell.
//! Precondition: `|element| < 2^31`, so that `float(element)` fits Q32.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[42, 10])?;
    let element = inputs.int("element")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let element = q.load_int(ctx, element);
    let fill = q.int_to_float(ctx, element);
    let mut result = Vec::with_capacity(93 * 13);
    for row in a.chunks(10) {
        result.extend_from_slice(row);
        result.extend([fill; 3]);
    }
    result.extend([fill; 51 * 13]);
    out.floats(&result);
    Ok(())
}
