// cholesky: literal translation of kernel.py (LITERAL_BASELINES.md).
// Empty dots are constant zero; nonempty dots rescale once; fresh divisor bound at every division.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let at = |i: usize, j: usize| i * N + j;
    // A[0, 0] = np.sqrt(A[0, 0])
    let root = q.sqrt(ctx, a[at(0, 0)]);
    a[at(0, 0)] = root.cell;


    for i in 1..N {
        for j in 0..i {
            // A[i, j] -= np.dot(A[i, :j], A[j, :j])
            let dot = if j > 0 {
                let dot = q.dot(ctx, &a[at(i, 0)..at(i, j)], &a[at(j, 0)..at(j, j)]);
                dot.cell
            } else { q.float_const(ctx, 0.0) };
            a[at(i, j)] = q.sub(ctx, a[at(i, j)], dot);
            // A[i, j] /= A[j, j]
            let divisor = q.bound(ctx, a[at(j, j)]);
            a[at(i, j)] = q.div(ctx, a[at(i, j)], divisor).cell;
        }
        // A[i, i] -= np.dot(A[i, :i], A[i, :i])
        let dot = q.dot(ctx, &a[at(i, 0)..at(i, i)], &a[at(i, 0)..at(i, i)]);
        let radicand = q.sub(ctx, a[at(i, i)], dot.cell);
        // A[i, i] = np.sqrt(A[i, i])
        let root = q.sqrt(ctx, radicand);
        a[at(i, i)] = root.cell;
    
    }
    out.floats(&a);
    Ok(())
}
