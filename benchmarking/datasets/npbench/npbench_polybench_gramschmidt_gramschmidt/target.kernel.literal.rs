// npbench_polybench_gramschmidt_gramschmidt__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each dot rescales once; updates materialize full column products. Constant lower-triangle outputs remain omitted.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NR: usize = 14; // = A_0
const NC: usize = 12; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[NR, NC])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut a = q.load_floats(ctx, &a);
    let zero = q.float_const(ctx, 0.0);
    let mut result = vec![zero; NR * NC];
    let mut r = vec![zero; NC * NC];
    for k in 0..NC {
        let column: Vec<Cell> = (0..NR).map(|i| a[i * NC + k]).collect();
        let nrm = q.dot(ctx, &column, &column).cell;
        r[k * NC + k] = q.sqrt(ctx, nrm).cell;
        for i in 0..NR {
            let divisor = q.bound(ctx, r[k * NC + k]);
            result[i * NC + k] = q.div(ctx, a[i * NC + k], divisor).cell;
        }
        for j in k + 1..NC {
            let left: Vec<Cell> = (0..NR).map(|i| result[i * NC + k]).collect();
            let right: Vec<Cell> = (0..NR).map(|i| a[i * NC + j]).collect();
            r[k * NC + j] = q.dot(ctx, &left, &right).cell;
            let products: Vec<Cell> = (0..NR)
                .map(|i| q.mul(ctx, result[i * NC + k], r[k * NC + j]).cell)
                .collect();
            for i in 0..NR {
                a[i * NC + j] = q.sub(ctx, a[i * NC + j], products[i]);
            }
        }
    }
    out.floats(&result);
    for k in 0..NC {
        out.const_floats(&vec![0.0; k]);
        out.floats(&r[k * NC + k..(k + 1) * NC]);
    }
    Ok(())
}
