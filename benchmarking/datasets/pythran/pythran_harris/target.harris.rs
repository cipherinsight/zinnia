//! pythran_harris__harris (Pythran harris), halo2-base baseline.
//! Kernel: Harris response `det - k * tr * tr` of I: Float[M, N] with
//! `dx[i, j] = I[i + 1, j + 1] - I[i, j + 1]`,
//! `dy[i, j] = I[i + 1, j + 1] - I[i + 1, j]` (both `(M - 1) x (N - 1)`),
//! `A = dx * dx`, `B = dy * dy`, `C = dx * dy`, `tr = A + B`,
//! `det = A * B - C * C` and `k = 0.05`.
//! Every `*` is an elementwise product NumPy materializes, so each is rounded
//! (`k * tr * tr` is `(k * tr) * tr`). `I[0, 0]` reaches no output (the
//! slices drop it), so it is not loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 4; // = I_0
const N: usize = 6; // = I_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let image = inputs.floats("I", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // Every element but I[0, 0], row-major.
    let mut pixels: Vec<Option<Cell>> = vec![None; M * N];
    for (at, v) in image.iter().enumerate().skip(1) {
        pixels[at] = Some(q.load_float(ctx, *v));
    }
    let at = |i: usize, j: usize| pixels[i * N + j].expect("read pixel");
    let k = Constant(fe(quantize(0.05)));
    let mut result = Vec::with_capacity((M - 1) * (N - 1));
    for i in 0..M - 1 {
        for j in 0..N - 1 {
            let dx = q.sub(ctx, at(i + 1, j + 1), at(i, j + 1));
            let dy = q.sub(ctx, at(i + 1, j + 1), at(i + 1, j));
            let a = q.mul(ctx, dx, dx).cell;
            let b = q.mul(ctx, dy, dy).cell;
            let c = q.mul(ctx, dx, dy).cell;
            let tr = q.add(ctx, a, b);
            let ab = q.mul(ctx, a, b);
            let cc = q.mul(ctx, c, c);
            let det = q.sub(ctx, ab, cc);
            let k_tr = q.mul(ctx, k, tr);
            let k_tr_tr = q.mul(ctx, k_tr, tr);
            result.push(q.sub(ctx, det, k_tr_tr));
        }
    }
    out.floats(&result);
    Ok(())
}
