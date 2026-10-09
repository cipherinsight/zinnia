//! pythran_goodexpomeasure__goodExpoMeasure (Pythran goodexpomeasure),
//! halo2-base baseline.
//! Kernel: the good-exposure measure of `inRGB: Int[C, H, W]` (only the
//! planes 0, 1, 2 are read, and loaded) under `sigma != 0` (`@requires`; the
//! division has no witness for a zero divisor): per pixel
//! `e_c = np.exp(-(c - 128) ** 2 / sigma)` for the three planes,
//! `x = e_R * e_G * e_B`, and the output
//! `(np.round(x, 2) * (2 ** 8 - 1)).astype(np.uint8)`.
//! `c - 128` is an integer, so its square is exact; the quotient by `sigma`
//! is rounded (one sign test of `sigma` for all pixels, `q.div_by_abs`), and
//! `exp` is the shared gadget (`q.exp`; an argument outside the window has
//! no proof). The two products are rounded in NumPy's order.
//! Rounds the Q32 product to hundredths (ties to even), then truncates
//! `255*n/100` and wraps modulo 256, including negative-sigma inputs.
//! LIMITATION: this follows Q32 arithmetic, not exact float64 discrete
//! outcomes near rounding boundaries. Float64 `(n/100)*255` can also lie
//! just below an integer (e.g. n=12820); see BATCH4_VALIDATION.md.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const C: usize = 16; // = inRGB_0
const H: usize = 16; // = inRGB_1
const W: usize = 16; // = inRGB_2

/// The planes the kernel reads (R, G, B), and the centre value, from its text.
const PLANES: usize = 3;
const _: () = assert!(C >= PLANES, "IndexError");
const CENTRE: i64 = 128;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let in_rgb = inputs.ints("inRGB", &[C, H, W])?;
    let sigma = inputs.float("sigma")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let planes = q.load_ints(ctx, &in_rgb[..PLANES * H * W]);
    let sigma = q.load_float_bounded(ctx, sigma);
    let magnitude = q.abs(ctx, sigma);
    // e[c][p] = exp(-(plane_c[p] - 128) ** 2 / sigma).
    let e: Vec<Vec<Cell>> = planes
        .chunks(H * W)
        .map(|plane| {
            plane
                .iter()
                .map(|v| {
                    let d = q.sub(ctx, *v, Constant(fe(CENTRE as i128)));
                    let square = q.gate.mul(ctx, d, d);
                    // -(d ** 2) as a float: -d^2 2^32.
                    let numerator = q.gate.mul(ctx, square, Constant(-fe(SCALE)));
                    let t = q.div_by_abs(ctx, numerator, sigma.cell, magnitude);
                    q.exp(ctx, t).cell
                })
                .collect()
        })
        .collect();
    let mut result: Vec<Cell> = Vec::with_capacity(H * W);
    for p in 0..H * W {
        let rg = q.mul(ctx, e[0][p], e[1][p]);
        let x = q.mul(ctx, rg, e[2][p]);
        let n = rint_hundredths(&q, ctx, x.cell);
        result.push(to_byte(&q, ctx, n));
    }
    out.ints(&result);
    Ok(())
}

/// `rint(100 x)` for a Q32 `x`, for a nonnegative Q32 product. Hints `m`, `f`: `100 x + 2^31 = m 2^32 + f` with `f` in
/// `[0, 2^32)` and `m` in `[0, 2^38)` (unique: the right side is below `2^70`),
/// so `m = floor(100 x + 1/2)`; at a tie (`f = 0`) an odd `m` steps down to
/// the even neighbour (`m = 2 h + odd`, `h` in `[0, 2^37)`).
fn rint_hundredths(q: &Q32, ctx: &mut Ctx, x: Cell) -> Cell {
    let shifted = q.gate.mul_add(ctx, x, Constant(Fr::from(100)), Constant(Fr::from(1u64 << 31)));
    let v = crate::q32::signed(shifted.value());
    // Hints, pinned by the decomposition and parity checks in rint_with.
    let m = v.div_euclid(SCALE);
    rint_with(q, ctx, shifted, fe(m), fe(v.rem_euclid(SCALE)), fe(m >> 1), fe(m & 1))
}

fn rint_with(q: &Q32, ctx: &mut Ctx, shifted: Cell, mh: Fr, fh: Fr, hh: Fr, oddh: Fr) -> Cell {
    // Hints are checked below; tests supply malicious alternatives here.
    let (m, f) = (ctx.load_witness(mh), ctx.load_witness(fh));
    q.range.range_check(ctx, m, 38);
    q.range.range_check(ctx, f, 32);
    let recomposed = q.gate.mul_add(ctx, m, Constant(fe(SCALE)), f);
    ctx.constrain_equal(&recomposed, &shifted);
    let (h, odd) = (ctx.load_witness(hh), ctx.load_witness(oddh));
    q.gate.assert_bit(ctx, odd);
    q.range.range_check(ctx, h, 37);
    let parity = q.gate.mul_add(ctx, h, Constant(Fr::from(2)), odd);
    ctx.constrain_equal(&parity, &m);
    let tie = q.gate.is_zero(ctx, f);
    let down = q.gate.mul(ctx, tie, odd);
    let n = q.sub(ctx, m, down);

    n
}

/// Truncate the nonnegative scaled value and wrap modulo 256, as uint8.
/// n < 2^38, so 255*n < 2^46 and its quotient is below 2^40.
fn to_byte(q: &Q32, ctx: &mut Ctx, n: Cell) -> Cell {
    let scaled = q.gate.mul(ctx, n, Constant(Fr::from(255)));
    let integer = q.divmod_small(ctx, scaled, 100, 46).0;
    q.divmod_small(ctx, integer, 256, 40).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_base::gates::circuit::CircuitBuilderStage;
    use halo2_base::halo2_proofs::dev::MockProver;

    fn accepts(f: impl FnOnce(&Q32, &mut Ctx)) -> bool {
        let mut builder = Builder::from_stage(CircuitBuilderStage::Mock);
        builder.set_k(12);
        builder.set_lookup_bits(8);
        builder.set_instance_columns(1);
        let q = Q32::new(&builder);
        f(&q, builder.main(0));
        builder.calculate_params(Some(20));
        let ok = MockProver::run(12, &builder, vec![vec![]]).unwrap().verify().is_ok();
        builder.clear();
        ok
    }

    #[test]
    fn hundredths_rejects_false_split_and_parity() {
        let run = |m, f, h, odd| accepts(|q, ctx| {
            let shifted = ctx.load_witness(fe(13 * SCALE));
            rint_with(q, ctx, shifted, fe(m), fe(f), fe(h), fe(odd));
        });
        assert!(run(13, 0, 6, 1));
        assert!(!run(12, SCALE, 6, 0));
        assert!(!run(14, -SCALE, 7, 0));
        assert!(!run(13, 0, 6, 0));
        assert!(!run(13, 0, 7, -1));
    }

    #[test]
    fn hundredths_ties_even_and_byte_wrap() {
        assert!(accepts(|q, ctx| {
            for (code, expected) in [(SCALE / 8, 12), (3 * SCALE / 8, 38)] {
                let x = ctx.load_witness(fe(code));
                let n = rint_hundredths(q, ctx, x);
                q.gate.assert_is_const(ctx, &n, &fe(expected));
            }
            for (n, expected) in [(100, 255), (101, 1), (200, 254), (1000, 246)] {
                let n = ctx.load_witness(fe(n));
                let byte = to_byte(q, ctx, n);
                q.gate.assert_is_const(ctx, &byte, &fe(expected));
            }
        }));
    }
}
