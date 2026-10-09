//! pythran_lapl3d__laplacien (Pythran lapl3d), halo2-base baseline.
//! Kernel: on image: Float[H, W, C], the absolute 5-point Laplacian of the
//! interior, `out = |4 c - up - down - left - right|` (`(H - 2) x (W - 2) x C`),
//! divided by `max(1.0, np.max(out)) + 1e-9`.
//! The stencil is exact (one inner product with constant weights); `np.abs`
//! is `q.abs_int64`. The maximum is `q.max_of` over the magnitudes and the
//! constant 1.0, which is `max(1.0, np.max(out))` in one hint. Every
//! magnitude is below the divisor, so each quotient lies in `[0, 1)`:
//! `q.div_fraction`, whose quotient check is 32 bits instead of the window
//! (122,194 advice and 30,172 lookup cells with `q.div_by_abs`). The four
//! corner pixels reach no output, so they are not loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const H: usize = 16; // = image_0
const W: usize = 16; // = image_1
const C: usize = 3; // = image_2

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let image = inputs.floats("image", &[H, W, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let corner = |i: usize, j: usize| (i == 0 || i == H - 1) && (j == 0 || j == W - 1);
    let mut pixels: Vec<Option<Cell>> = vec![None; H * W * C];
    for i in 0..H {
        for j in 0..W {
            if !corner(i, j) {
                for k in 0..C {
                    let at = (i * W + j) * C + k;
                    pixels[at] = Some(q.load_float(ctx, image[at]));
                }
            }
        }
    }
    let px = |i: usize, j: usize, k: usize| Existing(pixels[(i * W + j) * C + k].expect("read pixel"));
    let weights = [4, -1, -1, -1, -1].map(|w| Constant(fe(w)));
    let mut magnitudes = Vec::with_capacity((H - 2) * (W - 2) * C);
    for i in 1..H - 1 {
        for j in 1..W - 1 {
            for k in 0..C {
                let stencil = [px(i, j, k), px(i - 1, j, k), px(i + 1, j, k), px(i, j - 1, k), px(i, j + 1, k)];
                let laplacian = q.gate.inner_product(ctx, stencil, weights);
                magnitudes.push(q.abs_int64(ctx, laplacian));
            }
        }
    }
    let one = ctx.load_constant(fe(SCALE));
    let candidates: Vec<Cell> = magnitudes.iter().copied().chain([one]).collect();
    let largest = q.max_of(ctx, &candidates);
    let divisor = q.add(ctx, largest, Constant(fe(quantize(1e-9))));
    let result: Vec<Cell> = magnitudes.iter().map(|m| q.div_fraction(ctx, *m, divisor)).collect();
    out.floats(&result);
    Ok(())
}
