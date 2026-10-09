// pythran_periodic_bc__periodic_bc: literal translation of kernel.py (LITERAL_BASELINES.md).
// Slice copy; overwritten unread input plane gets no domain check, as required by statement plumbing.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N0: usize = 3; // = f_0
const N1: usize = 3; // = f_1
const N2: usize = 3; // = f_2
/// The plane `-2` names (NumPy's wrap of a negative index).
const SOURCE: usize = N1 - 2;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let f = inputs.floats("f", &[N0, N1, N2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let zero = q.float_const(ctx, 0.0);
    let mut f_cells = vec![zero; N0 * N1 * N2];
    for i in 0..N0 {
        for j in 0..N1 {
            if j != 0 || SOURCE == 0 {
                for k in 0..N2 {
                    f_cells[(i * N1 + j) * N2 + k] = q.load_float(ctx, f[(i * N1 + j) * N2 + k]);
                }
            }
        }
    }
    let source: Vec<Cell> = (0..N0)
        .flat_map(|i| (0..N2).map(move |k| (i, k)))
        .map(|(i, k)| f_cells[(i * N1 + SOURCE) * N2 + k])
        .collect();
    for i in 0..N0 {
        for k in 0..N2 {
            f_cells[i * N1 * N2 + k] = source[i * N2 + k];
        }
    }
    out.floats(&f_cells);
    Ok(())
}
