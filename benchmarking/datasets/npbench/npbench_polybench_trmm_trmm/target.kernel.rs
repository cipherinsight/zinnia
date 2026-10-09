//! npbench_polybench_trmm_trmm__kernel (NPBench polybench/trmm), halo2-base baseline.
//! Kernel: on alpha: Float, A: Float[NI, NI], B: Float[NI, NJ], in place:
//! `B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])` for every `i`, `j`, then
//! `B *= alpha`; the output is `B` after the call (`mutated_outputs`).
//! Iteration `i` reads rows below `i` of `B`, which later iterations write,
//! so every dot product reads the original `B`. Each is one reduction,
//! rescaled once (`q.dot`; the last row's is empty, 0.0); each element of
//! `B *= alpha` is a rounded product. Only the strict lower triangle of `A`
//! is read, and only it is loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 13; // = A_0
const NJ: usize = 16; // = B_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let a = inputs.floats("A", &[NI, NI])?;
    let b = inputs.floats("B", &[NI, NJ])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    // column[i] = A[i + 1:, i]
    let columns: Vec<Vec<Cell>> =
        (0..NI).map(|i| (i + 1..NI).map(|k| q.load_float(ctx, a[k * NI + i])).collect()).collect();
    let b = q.load_floats(ctx, &b);
    let mut result = Vec::with_capacity(NI * NJ);
    for (i, column) in columns.iter().enumerate() {
        for j in 0..NJ {
            let updated = if column.is_empty() {
                b[i * NJ + j]
            } else {
                let below: Vec<Cell> = (i + 1..NI).map(|k| b[k * NJ + j]).collect();
                let dot = q.dot(ctx, column, &below);
                q.add(ctx, b[i * NJ + j], dot.cell)
            };
            result.push(q.mul(ctx, updated, alpha).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
