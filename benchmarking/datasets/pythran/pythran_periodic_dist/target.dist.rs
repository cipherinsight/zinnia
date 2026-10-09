//! Source periodic distance computation for every ordered particle pair.
//! Each enabled axis performs its positive-half-box masked subtraction,
//! then compares the UPDATED value for its negative-half-box addition.
//! Each directed entry is squared independently; no symmetry identities.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let coordinates = ["x", "y", "z"].map(|name| inputs.floats(name, &[N]));
    let l = inputs.int("L")?;
    let flags = ["periodicX", "periodicY", "periodicZ"].map(|name| inputs.bools(name, &[]));
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let l = q.load_int(ctx, l);
    let mut axes = Vec::new();
    for (coordinates, flags) in coordinates.into_iter().zip(flags) {
        let c = q.load_floats(ctx, &coordinates?);
        let p = q.load_bools(ctx, &flags?)[0];
        // Inactive scalar branch uses zero, avoiding dead-branch conversion
        // overflow. Every enabled branch uses the original private L.
        let enabled_l = q.gate.mul(ctx, p, l);
        let l_float = q.int_to_float(ctx, enabled_l);
        let half = q.gate.mul(ctx, enabled_l, Constant(fe(SCALE / 2)));
        let half = q.bound(ctx, half);
        let negative_half = q.neg(ctx, half.cell);
        let negative_half = q.bound(ctx, negative_half);
        let mut entries = Vec::with_capacity(N * N);
        for i in 0..N {
            for j in 0..N {
                let delta = q.sub(ctx, c[j], c[i]);
                let delta = q.bound(ctx, delta);
                let above = q.gt(ctx, delta, half);
                let subtract = q.gate.and(ctx, p, above);
                let less_l = q.sub(ctx, delta.cell, l_float);
                let updated = q.gate.select(ctx, less_l, delta.cell, subtract);
                let updated = q.bound(ctx, updated);
                let below = q.lt(ctx, updated, negative_half);
                let add = q.gate.and(ctx, p, below);
                let plus_l = q.add(ctx, updated.cell, l_float);
                entries.push(q.gate.select(ctx, plus_l, updated.cell, add));
            }
        }
        axes.push(entries);
    }
    let mut distances = Vec::with_capacity(N * N);
    for at in 0..N * N {
        let sx = q.mul(ctx, axes[0][at], axes[0][at]);
        let sy = q.mul(ctx, axes[1][at], axes[1][at]);
        let sz = q.mul(ctx, axes[2][at], axes[2][at]);
        let sum = q.add(ctx, sx.cell, sy.cell);
        let sum = q.add(ctx, sum, sz.cell);
        let root = q.sqrt(ctx, sum);
        let zero = q.gate.is_zero(ctx, root.cell);
        distances.push(q.gate.select(ctx, Constant(fe(-SCALE)), root.cell, zero));
    }
    out.floats(&distances);
    for axis in axes { out.floats(&axis); }
    Ok(())
}
