//! DS-1000 case 0362, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel:
//!   del_col = np.array([12, 14, 1, 9])
//!   mask = (del_col <= a.shape[1])
//!   del_col = del_col[mask] - 1
//!   result = np.delete(a, del_col, axis=1)
//! on `Float[8, 6]`. Every value in these four lines is a Python-level
//! constant (`del_col`'s literal entries, `a.shape[1] = 6`), so the whole
//! computation is compile-time (L3: arithmetic on compile-time numbers); the
//! circuit does no work beyond loading and exposing the kept input cells.
//! Evaluating the four lines in order: `mask = [12<=6, 14<=6, 1<=6, 9<=6] =
//! [false, false, true, false]`, `del_col[mask] - 1 = [0]`, so column 0 is
//! deleted and columns 1..6 are kept.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // del_col = np.array([12, 14, 1, 9])
    let del_col = [12i64, 14, 1, 9];
    // mask = (del_col <= a.shape[1])
    let shape1: i64 = 6;
    let mask: Vec<bool> = del_col.iter().map(|&v| v <= shape1).collect();
    // del_col = del_col[mask] - 1
    let del_col: Vec<usize> = del_col
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&v, _)| (v - 1) as usize)
        .collect();
    // result = np.delete(a, del_col, axis=1)
    let kept: Vec<f64> = (0..8)
        .flat_map(|i| (0..6).filter(|j| !del_col.contains(j)).map(move |j| a[i * 6 + j]))
        .collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
