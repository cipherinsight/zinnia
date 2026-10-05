//! DS-1000 case 0478, halo2-base baseline.
//! Kernel: `result = np.einsum('ii->i', a); save = result.copy(); a[...] = 0;
//! result[...] = save; return a` on `Integer[4, 4]`.
//! `np.einsum('ii->i', a)` is a writeable view of `a`'s diagonal, so the last
//! assignment writes the diagonal back into the zeroed `a`: the output is the
//! diagonal of `a` with zeros elsewhere. The zeros are circuit constants, not
//! public cells; only the 4 diagonal cells are loaded and exposed.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let diag = q.load_ints(ctx, &[a[0], a[5], a[10], a[15]]);
    for (i, d) in diag.iter().enumerate() {
        if i > 0 {
            out.const_ints(&[0; 4]); // a[i - 1, i:], a[i, :i]
        }
        out.int(*d);
    }
    Ok(())
}
