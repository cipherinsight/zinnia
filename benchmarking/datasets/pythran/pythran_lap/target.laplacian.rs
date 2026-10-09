//! pythran_lap__laplacian (Pythran lap), halo2-base baseline.
//! Kernel: the 4th-order periodic 1-D Laplacian of `var: Float[N]` built by
//! nine slice updates of `lap` (with the kernel's quirk `lap[0] = 4/3 *
//! var[1]`, not `var[-1]`), then `lap / dh2`, under `dh2 != 0.0`
//! (`@requires`; the division has no witness for a zero divisor).
//! Each update adds a constant times a slice of `var`; NumPy materializes the
//! products, so each is rounded (`q.mul`), and a product of the same constant
//! and element is computed once. The divisor's sign test is hoisted out of
//! the elementwise division (`q.abs` once, then `q.div_by_abs`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = var_0

const NEAR: f64 = 4.0 / 3.0;
const CENTER: f64 = -5.0 / 2.0;
const FAR: f64 = -1.0 / 12.0;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let var = inputs.floats("var", &[N])?;
    let dh2 = inputs.float("dh2")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let var = q.load_floats(ctx, &var);
    let dh2 = q.load_float_bounded(ctx, dh2);
    let mut scaled = |c: f64| -> Vec<Cell> { var.iter().map(|v| q.mul(ctx, *v, Constant(fe(quantize(c)))).cell).collect() };
    let (near, center, far) = (scaled(NEAR), scaled(CENTER), scaled(FAR));

    // The terms each element of `lap` collects, in the kernel's order.
    let mut terms: Vec<Vec<Cell>> = vec![Vec::new(); N];
    for i in 1..N {
        terms[i].push(near[i - 1]); // lap[1:] = 4/3 * var[:-1]
    }
    terms[0].push(near[1]); // lap[0] = 4/3 * var[1]
    for i in 0..N - 1 {
        terms[i].push(near[i + 1]); // lap[:-1] += 4/3 * var[1:]
    }
    terms[N - 1].push(near[0]); // lap[-1] += 4/3 * var[0]
    for i in 0..N {
        terms[i].push(center[i]); // lap += -5/2 * var
    }
    for i in 2..N {
        terms[i].push(far[i - 2]); // lap[2:] += -1/12 * var[:-2]
    }
    for i in 0..2 {
        terms[i].push(far[N - 2 + i]); // lap[:2] += -1/12 * var[-2:]
    }
    for i in 0..N - 2 {
        terms[i].push(far[i + 2]); // lap[:-2] += -1/12 * var[2:]
    }
    for i in 0..2 {
        terms[N - 2 + i].push(far[i]); // lap[-2:] += -1/12 * var[:2]
    }

    let magnitude = q.abs(ctx, dh2);
    let result: Vec<Cell> = terms
        .into_iter()
        .map(|t| {
            let lap = q.sum(ctx, t);
            q.div_by_abs(ctx, lap, dh2.cell, magnitude).cell
        })
        .collect();
    out.floats(&result);
    Ok(())
}
