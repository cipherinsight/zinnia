//! pythran_pythagorean_triples__next_pythagorean_triples (Pythran pythagorean_triples), halo2-base baseline.
//! Kernel: the three children of each Pythagorean triple in
//! `previous: Int[P, 3]` (Berggren's matrices, a constant 9 x 3 integer
//! matrix of the kernel, stacked), reshaped to `3 P` rows of 3.
//! Output row `3 p + t` is matrices `3 t .. 3 t + 2` applied to
//! `previous[p]`: each value is a constant-coefficient integer combination of
//! the triple, exact in the field (no rescale, no range check).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const P: usize = 1; // = previous_0
const COLS: usize = 3; // = previous_1

/// Berggren's matrices `A`, `B`, `C`, stacked (the kernel's constant).
const MATRICES: [[i64; 3]; 9] = [
    [-1, 2, 2], [-2, 1, 2], [-2, 2, 3],
    [1, 2, 2], [2, 1, 2], [2, 2, 3],
    [1, -2, 2], [2, -1, 2], [2, -2, 3],
];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let previous = inputs.ints("previous", &[P, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let previous = q.load_ints(ctx, &previous);
    let mut result = Vec::with_capacity(P * MATRICES.len());
    for triple in previous.chunks(COLS) {
        for row in &MATRICES {
            let coefficients = row.iter().map(|c| Constant(fe(*c as i128)));
            result.push(q.gate.inner_product(ctx, triple.iter().map(|x| Existing(*x)), coefficients));
        }
    }
    out.ints(&result);
    Ok(())
}
