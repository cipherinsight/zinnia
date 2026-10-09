//! pythran_specialconvolve__specialconvolve (Pythran specialconvolve), halo2-base baseline.
//! Kernel: on `a: Int[R, C]`, `rowconvol` sums each 3-row window
//! (`[R - 2, C]`), `colconvol` sums each 3-column window of it and subtracts
//! 9 times the window's centre (`[R - 2, C - 2]`, the output).
//! Integer sums, exact in the field. The row sums are computed once and
//! shared by the three column windows that use them (3,528 cells of
//! arithmetic at the fixture's shape, against 4,900 for one 9-term
//! combination per output).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const R: usize = 16; // = a_0
const C: usize = 16; // = a_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[R, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let at = |i: usize, j: usize| a[i * C + j];
    // rowconvol = a[1:-1, :] + a[:-2, :] + a[2:, :]
    let rowconvol: Vec<Cell> = (0..R - 2)
        .flat_map(|i| (0..C).map(move |j| (i, j)))
        .map(|(i, j)| q.sum(ctx, [at(i, j), at(i + 1, j), at(i + 2, j)]))
        .collect();
    let rc = |i: usize, j: usize| rowconvol[i * C + j];
    // colconvol = rowconvol[:, 1:-1] + rowconvol[:, :-2] + rowconvol[:, 2:] - 9 * a[1:-1, 1:-1]
    let mut colconvol = Vec::with_capacity((R - 2) * (C - 2));
    for i in 0..R - 2 {
        for j in 0..C - 2 {
            let terms = [rc(i, j), rc(i, j + 1), rc(i, j + 2), at(i + 1, j + 1)].map(Existing);
            let weights = [1, 1, 1, -9].map(|w| Constant(fe(w)));
            colconvol.push(q.gate.inner_product(ctx, terms, weights));
        }
    }
    out.ints(&colconvol);
    Ok(())
}
