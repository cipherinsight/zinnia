//! pythran_pairwise__pairwise: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compute all ordered pairs including diagonal; omit constant diagonal only from public outputs.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 16; // = X_0
const N: usize = 8; // = X_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("X", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &raw);
    let zero = q.float_const(ctx, 0.0);
    let mut distances = vec![zero; M * M];
    for i in 0..M {
        for j in 0..M {
            let mut d = q.float_const(ctx, 0.0);
            for k in 0..N {
                let tmp = q.sub(ctx, x[i * N + k], x[j * N + k]);
                let product = q.mul(ctx, tmp, tmp).cell;
                d = q.add(ctx, d, product);
            }
            q.range.range_check(ctx, d, 63);
            distances[i * M + j] = q.sqrt(ctx, d).cell;
        }
    }
    for i in 0..M {
        for j in 0..M {
            if i == j {
                out.const_floats(&[0.0]);
            } else {
                out.float(distances[i * M + j]);
            }
        }
    }
    Ok(())
}
