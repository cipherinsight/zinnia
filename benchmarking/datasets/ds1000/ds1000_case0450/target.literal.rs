//! DS-1000 case 0450: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Broadcast gather is evaluated axis by axis, with standard q.index calls.
use crate::prelude::*;

const D0: usize = 7;
const D1: usize = 8;
const D2: usize = 9;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[D0, D1, D2])?;
    let (second, third) = (inputs.ints("second", &[2])?, inputs.ints("third", &[2])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let second = q.load_ints(ctx, &second);
    let third = q.load_ints(ctx, &third);

    // result[i, j, k] = a[i, second[j], third[k]]
    let mut result = Vec::with_capacity(D0 * 2 * 2);
    for i in 0..D0 {
        for j in 0..2 {
            // row = a[i, second[j], :]
            let row: Vec<Cell> = (0..D2)
                .map(|p2| {
                    let candidates: Vec<Cell> =
                        (0..D1).map(|p1| a[(i * D1 + p1) * D2 + p2]).collect();
                    q.index(ctx, &candidates, second[j])
                })
                .collect();
            for k in 0..2 {
                result.push(q.index(ctx, &row, third[k]));
            }
        }
    }
    out.floats(&result);
    Ok(())
}
