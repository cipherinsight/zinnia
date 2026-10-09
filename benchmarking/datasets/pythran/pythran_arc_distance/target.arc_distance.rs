//! pythran_arc_distance__arc_distance (Pythran arc_distance), halo2-base baseline.
//! Kernel: the haversine distance of points (theta_1, phi_1) and
//! (theta_2, phi_2), each Float[N]:
//! `temp = sin((theta_2 - theta_1) / 2) ** 2 + cos(theta_1) * cos(theta_2) * sin((phi_2 - phi_1) / 2) ** 2`,
//! `2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))`.
//! `v / 2` is `floor(v / 2)` (`q.floor_div_const`); `sin`, `cos` are the
//! library's; every `*` and `** 2` is a rounded product, left to right; the
//! square roots are exact floors, and as proven non-negative values they feed
//! `q.arctan2_nonneg` (no sign handling). Where the rounded `temp` falls
//! outside `[0, 1]` (possible only within a few units of 2^-32 of an end, as
//! float64's `temp` can, which makes NumPy's result NaN) a square root has no
//! witness.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 5; // = theta_1_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let theta_1 = inputs.floats("theta_1", &[N])?;
    let phi_1 = inputs.floats("phi_1", &[N])?;
    let theta_2 = inputs.floats("theta_2", &[N])?;
    let phi_2 = inputs.floats("phi_2", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let theta_1 = q.load_floats(ctx, &theta_1);
    let phi_1 = q.load_floats(ctx, &phi_1);
    let theta_2 = q.load_floats(ctx, &theta_2);
    let phi_2 = q.load_floats(ctx, &phi_2);
    let half_angle_sin_sq = |ctx: &mut Ctx, from: Cell, to: Cell| {
        let diff = q.sub(ctx, to, from);
        let (half, _) = q.floor_div_const(ctx, diff, 2);
        let s = q.sin(ctx, half);
        q.mul(ctx, s, s).cell
    };
    let mut result = Vec::with_capacity(N);
    for i in 0..N {
        let lat = half_angle_sin_sq(ctx, theta_1[i], theta_2[i]);
        let lon = half_angle_sin_sq(ctx, phi_1[i], phi_2[i]);
        let c1 = q.cos(ctx, theta_1[i]);
        let c2 = q.cos(ctx, theta_2[i]);
        let cc = q.mul(ctx, c1, c2);
        let weighted = q.mul(ctx, cc, lon);
        let temp = q.add(ctx, lat, weighted);
        let rest = q.sub(ctx, Constant(fe(SCALE)), temp);
        let y = q.sqrt(ctx, temp);
        let x = q.sqrt(ctx, rest);
        let angle = q.arctan2_nonneg(ctx, y.cell, x.cell);
        result.push(q.add(ctx, angle, angle));
    }
    out.floats(&result);
    Ok(())
}
