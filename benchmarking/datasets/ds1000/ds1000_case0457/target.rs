//! DS-1000 case 0457, optimized halo2-base baseline.
//! dim = 6 is the input-width constant; the source body never reads it.
//! Pairwise Euclidean norms: compute each unordered pair once and mirror it.
//! One fused sum-of-squares rescale per norm, then the shared floor sqrt.
//! The rescaled squared norm is in the Q32 window; both variants impose this.
//! Diagonal zeros are compile-time outputs, not public instances.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let zero = q.float_const(ctx, 0.0);
    let mut distance = vec![zero; 64];
    for i in 0..8 {
        for j in i + 1..8 {
            let d: Vec<_> = (0..6).map(|k| q.sub(ctx, a[j * 6 + k], a[i * 6 + k])).collect();
            let squared = q.dot(ctx, &d, &d);
            let norm = q.sqrt(ctx, squared.cell).cell;
            distance[i * 8 + j] = norm;
            distance[j * 8 + i] = norm;
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            if i == j { out.const_floats(&[0.0]); }
            else { out.float(distance[i * 8 + j]); }
        }
    }
    Ok(())
}
