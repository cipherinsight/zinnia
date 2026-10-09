//! pythran_spectraldns__cross2 (Pythran spectraldns), halo2-base baseline.
//! Kernel: `cross1(c, a, b)` writes the cross product of the real vector
//! field a: Float[3, A1, A2, A3] and the complex one b: Complex[3, A1, A2, A3]
//! into c, `c[0] = a[0] * b[2] - a[2] * b[1]`, `c[1] = a[2] * b[0] - a[0] * b[2]`,
//! `c[2] = a[0] * b[1] - a[1] * b[0]`; then `c *= 1j`; returns `c`.
//! The leading 3 is the cross product's, not a size. `c` is overwritten
//! before it is read, so it is not loaded.
//! A real times a complex value is two rounded products (the real and
//! imaginary parts of the complex product with a zero imaginary part, each
//! one rescale); NumPy materializes each product array before subtracting, so
//! the products are rounded one by one, and `a[0] * b[2]`, used twice, is
//! computed once. `* 1j` maps `x + yi` to `-y + xi` exactly.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.cross2.json.
const A1: usize = 2; // = a_1
const A2: usize = 5; // = a_2
const A3: usize = 7; // = a_3
const POINTS: usize = A1 * A2 * A3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[3, A1, A2, A3])?;
    let b = inputs.complexes("b", &[3, A1, A2, A3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b); // interleaved real and imaginary parts
                                    // Component k of a and b at point p.
    let a_at = |k: usize, p: usize| a[k * POINTS + p];
    let b_at = |k: usize, p: usize| [b[2 * (k * POINTS + p)], b[2 * (k * POINTS + p) + 1]];
    let mut c = vec![[None::<Cell>; 2]; 3 * POINTS];
    for p in 0..POINTS {
        let product = |ctx: &mut Ctx, k: usize, l: usize| b_at(l, p).map(|part| q.mul(ctx, a_at(k, p), part).cell);
        let a0b2 = product(ctx, 0, 2);
        let a2b1 = product(ctx, 2, 1);
        let a2b0 = product(ctx, 2, 0);
        let a0b1 = product(ctx, 0, 1);
        let a1b0 = product(ctx, 1, 0);
        for (k, (plus, minus)) in [(a0b2, a2b1), (a2b0, a0b2), (a0b1, a1b0)].into_iter().enumerate() {
            // (re + im i) * 1j = -im + re i
            let minus_im = q.sub(ctx, minus[1], plus[1]);
            let re = q.sub(ctx, plus[0], minus[0]);
            c[k * POINTS + p] = [Some(minus_im), Some(re)];
        }
    }
    let c: Vec<Cell> = c.into_iter().flatten().map(|v| v.expect("every element written")).collect();
    out.floats(&c);
    Ok(())
}
