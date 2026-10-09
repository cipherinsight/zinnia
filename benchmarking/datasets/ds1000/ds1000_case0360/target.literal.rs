//! DS-1000 case 0360, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel: `a = np.delete(a, 2, axis=0)` on `Float[4, 4]`. The deleted row is
//! a Python-level constant (L3: compile-time numbers), so which cells are
//! kept is resolved before any witness is loaded; the circuit does no work
//! beyond loading and exposing the kept input cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // a = np.delete(a, 2, axis=0)
    let del_row: usize = 2;
    let kept: Vec<f64> = (0..4)
        .filter(|&i| i != del_row)
        .flat_map(|i| a[i * 4..i * 4 + 4].to_vec())
        .collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
