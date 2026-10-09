//! DS-1000 case 0361, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel:
//!   temp = np.array([0, 2])
//!   a = np.delete(a, temp, axis=1)
//! on `Float[6, 6]`. `temp` is a Python-level array literal (L3: compile-time
//! numbers), so the deleted columns are resolved before any witness is
//! loaded; the circuit does no work beyond loading and exposing the kept
//! input cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[6, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // temp = np.array([0, 2])
    let temp = [0usize, 2];
    // a = np.delete(a, temp, axis=1)
    let a = a.as_slice();
    let kept: Vec<f64> = (0..6)
        .flat_map(|i| (0..6).filter(|j| !temp.contains(j)).map(move |j| a[i * 6 + j]))
        .collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
