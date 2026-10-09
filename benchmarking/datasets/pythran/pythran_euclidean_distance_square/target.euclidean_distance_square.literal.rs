//! pythran_euclidean_distance_square__euclidean_distance_square: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each dot and square sum rescales once as optimized; full square arrays and source addition order remain.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const R1: usize = 1; // = x1_0
const R2: usize = 5; // = x2_0
const D: usize = 3; // = x1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("x1", &[R1, D])?;
    let b = inputs.floats("x2", &[R2, D])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x1 = q.load_floats(ctx, &a);
    let x2 = q.load_floats(ctx, &b);
    let mut gram = Vec::new();
    for i in 0..R1 {
        for j in 0..R2 {
            gram.push(
                q.dot(ctx, &x1[i * D..(i + 1) * D], &x2[j * D..(j + 1) * D])
                    .cell,
            );
        }
    }
    let scaled: Vec<Cell> = gram
        .iter()
        .map(|v| q.mul_int(ctx, *v, Constant(fe(-2))))
        .collect();
    let squares1: Vec<Cell> = x1.iter().map(|v| q.gate.mul(ctx, *v, *v)).collect();
    let sums1: Vec<Cell> = squares1
        .chunks(D)
        .map(|row| {
            let mut acc = q.float_const(ctx, 0.0);
            for v in row {
                acc = q.add(ctx, acc, *v);
            }
            q.rescale(ctx, acc).cell
        })
        .collect();
    let first: Vec<Cell> = (0..R1 * R2)
        .map(|i| q.add(ctx, scaled[i], sums1[i / R2]))
        .collect();
    let squares2: Vec<Cell> = x2.iter().map(|v| q.gate.mul(ctx, *v, *v)).collect();
    let sums2: Vec<Cell> = squares2
        .chunks(D)
        .map(|row| {
            let mut acc = q.float_const(ctx, 0.0);
            for v in row {
                acc = q.add(ctx, acc, *v);
            }
            q.rescale(ctx, acc).cell
        })
        .collect();
    let result: Vec<Cell> = (0..R1 * R2)
        .map(|i| q.add(ctx, first[i], sums2[i % R2]))
        .collect();
    out.floats(&result);
    Ok(())
}
