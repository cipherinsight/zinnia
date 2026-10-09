//! Source goodExpo sequence with canonical exp, per-channel products,
//! np.round(...,2) materialized in Q32, multiplication by255, truncation,
//! then uint8 wrap. No rational floor(255*n/100) shortcut.
//! Integer input conversion uses Q32, as the shared compiler arithmetic.
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
                    let converted = q.int_to_float(ctx, *v);
                    let d = q.sub(ctx, converted, Constant(fe(CENTRE as i128 * SCALE)));
                    // SubF discards the compiler's float-cast factor alias;
                    // its subsequent MulF rounds and checks the Q32 square.
                    let square = q.mul(ctx, d, d);
                    let numerator = q.neg(ctx, square.cell);
                    let t = q.div_by_abs(ctx, numerator, sigma.cell, magnitude);
                    q.canonical_exp(ctx, t).cell
                })
                .collect()
        })
        .collect();
    let mut result: Vec<Cell> = Vec::with_capacity(H * W);
    for p in 0..H * W {
        let rg = q.mul(ctx, e[0][p], e[1][p]);
        let x = q.mul(ctx, rg, e[2][p]);
        let n = rint_hundredths(&q, ctx, x.cell);
        let n_float = q.int_to_float(ctx, n);
        let rounded = q.div_const(ctx, n_float, 100.0);
        let scaled = q.mul(ctx, rounded, Constant(fe(255 * SCALE)));
        let integer = q.float_to_int(ctx, scaled);
        let (_, byte) = q.floor_div_const(ctx, integer.cell, 256);
        result.push(byte);
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
