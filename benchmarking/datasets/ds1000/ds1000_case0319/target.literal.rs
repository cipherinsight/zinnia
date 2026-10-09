//! DS-1000 case 0319: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Enforce permutation precondition; build inverse by private-index scatter, then standard indexed reads.
//! empty_like starts with zeros: the permutation guarantees every slot is overwritten.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6, 8])?;
    let permutation = inputs.ints("permutation", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let permutation = q.load_ints(ctx, &permutation);
    // @requires(is_permutation(permutation)): exactly one of each legal index.
    for j in 0..8 {
        let mut count = q.int_const(ctx, 0);
        for k in 0..8 {
            let hit = q.eq(ctx, permutation[k], Constant(fe(j as i128)));
            count = q.add(ctx, count, hit);
        }
        q.gate.assert_is_const(ctx, &count, &Fr::ONE);
    }
    // c = np.empty_like(permutation)
    let zero = q.int_const(ctx, 0);
    let mut c = vec![zero; 8];
    // c[permutation] = np.arange(len(permutation))
    let arange: Vec<Cell> = (0..8).map(|i| q.int_const(ctx, i as i64)).collect();
    for i in 0..8 {
        for k in 0..8 {
            let hit = q.eq(ctx, permutation[i], Constant(fe(k as i128)));
            c[k] = q.gate.select(ctx, arange[i], c[k], hit);
        }
    }
    let mut result = Vec::new();
    // result = a[c, :, :]
    for i in 0..8 {
        for j in 0..48 {
            let column: Vec<Cell> = (0..8).map(|k| a[k*48+j]).collect();
            result.push(q.index(ctx, &column, c[i]));
        }
    }
    out.floats(&result);
    Ok(())
}
