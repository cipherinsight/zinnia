//! DS-1000 case 0438, halo2-base baseline.
//! Kernel: `np.corrcoef(post, distance)[0][1]` on two `Integer[4]`.
//! Pearson r = Sxy / sqrt(Sxx * Syy) on centred sums (NumPy's population or
//! sample factors cancel). Scaled by n^2 = 16 the centred sums are the exact
//! integers A = 4 sum(xy) - sum(x) sum(y), B = 4 sum(x^2) - sum(x)^2 and
//! C = 4 sum(y^2) - sum(y)^2, so r = A / sqrt(B * C): one Q32 square root of the
//! integer B * C and one float division. |r| <= 1 holds exactly (Cauchy-Schwarz
//! on the integers, floor square root), so NumPy's clip to [-1, 1] never acts.
//! No proof when a variance is 0 (NumPy returns nan) or when B * C >= 2^62,
//! where sqrt(B * C) leaves the Q32 window (inputs within +-11585 always fit).
//! The inputs get window proofs, so A, B and C are exact integers (field
//! elements could otherwise make them any small integers, r = 3 included).
//! B and C are range-checked to 64 bits, so t = B * C * 2^64 < 2^192 is the
//! integer product (no field wrap) and the square root's checks
//! (y < 2^63, t - y^2 and y^2 + 2y - t in [0, 2^64)) give y = floor(sqrt(t))
//! and B * C < 2^62, Zinnia's domain. With B alone checked, B * C reaches
//! 2^194 and t can wrap the field to some t mod p < 2^126 whose root the checks
//! accept: a wrong r where NumPy has a value (`hw_check/domain_edges.json`,
//! 0438 `wrapped_root`: r = 13.7 for NumPy's 0.0). Checking C costs the same
//! as checking B * C < 2^62 at this case's lookup width (6 bits).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (post, distance) = (inputs.ints("post", &[4])?, inputs.ints("distance", &[4])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &post);
    let y = q.load_ints(ctx, &distance);
    let dot = |ctx: &mut Ctx, a: &[Cell], b: &[Cell]| {
        q.gate.inner_product(ctx, a.iter().map(|c| Existing(*c)), b.iter().map(|c| Existing(*c)))
    };
    let (sx, sy) = (q.sum(ctx, x.iter().copied()), q.sum(ctx, y.iter().copied()));
    let (sxy, sxx, syy) = (dot(ctx, &x, &y), dot(ctx, &x, &x), dot(ctx, &y, &y));
    // n * s - t * u with n = 4
    let centred = |ctx: &mut Ctx, s: Cell, t: Cell, u: Cell| {
        let ns = q.gate.mul(ctx, s, Constant(fe(4)));
        q.gate.sub_mul(ctx, ns, t, u)
    };
    let a = centred(ctx, sxy, sx, sy);
    let b = centred(ctx, sxx, sx, sx);
    let c = centred(ctx, syy, sy, sy);
    q.range.range_check(ctx, b, 64);
    q.range.range_check(ctx, c, 64);

    let bc = q.gate.mul(ctx, b, c);
    let bc = q.int_to_float(ctx, bc);
    let denom = q.sqrt(ctx, bc); // floor(sqrt(B * C) * 2^32); 0 has no division witness
    let a = q.int_to_float(ctx, a);
    let r = q.div_pos(ctx, a, denom);
    out.float(r.cell);
    Ok(())
}
