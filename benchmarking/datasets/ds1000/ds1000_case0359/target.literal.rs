//! DS-1000 case 0359, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel: `a = np.delete(a, 2, axis=1)` on `Float[3, 3]`. The deleted column
//! is a Python-level constant (L3: compile-time numbers), so which cells are
//! kept is resolved before any witness is loaded; the circuit does no work
//! beyond loading and exposing the kept input cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[3, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // a = np.delete(a, 2, axis=1)
    let del_col: usize = 2;
    let kept: Vec<f64> = (0..3)
        .flat_map(|i| (0..3).filter(|&j| j != del_col).map(move |j| a[i * 3 + j]))
        .collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
