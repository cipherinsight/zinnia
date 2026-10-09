// pythran_pythagorean_triples__next_pythagorean_triples: literal translation of kernel.py (LITERAL_BASELINES.md).
// Integer dot uses a running sum in contraction-index order; transposes and reshape move wires only.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const P: usize = 1; // = previous_0
const COLS: usize = 3; // = previous_1

/// Berggren's matrices `A`, `B`, `C`, stacked (the kernel's constant).
const MATRICES: [[i64; 3]; 9] = [
    [-1, 2, 2],
    [-2, 1, 2],
    [-2, 2, 3],
    [1, 2, 2],
    [2, 1, 2],
    [2, 2, 3],
    [1, -2, 2],
    [2, -1, 2],
    [2, -2, 3],
];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let previous = inputs.ints("previous", &[P, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let previous = q.load_ints(ctx, &previous);
    let zero = q.int_const(ctx, 0);
    let mut product = vec![zero; MATRICES.len() * P];
    for r in 0..MATRICES.len() {
        for p in 0..P {
            let mut acc = zero;
            for c in 0..COLS {
                let term = q.gate.mul(
                    ctx,
                    Constant(fe(MATRICES[r][c] as i128)),
                    previous[p * COLS + c],
                );
                acc = q.add(ctx, acc, term);
            }
            product[r * P + p] = acc;
        }
    }
    let result: Vec<Cell> = (0..P)
        .flat_map(|p| (0..MATRICES.len()).map(move |r| (p, r)))
        .map(|(p, r)| product[r * P + p])
        .collect();
    out.ints(&result);
    Ok(())
}
