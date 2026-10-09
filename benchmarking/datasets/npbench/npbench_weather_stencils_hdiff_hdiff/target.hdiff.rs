//! npbench_weather_stencils_hdiff_hdiff__hdiff (NPBench weather_stencils/hdiff),
//! halo2-base baseline.
//! Kernel: horizontal diffusion on `in_field: Float[I + 4, J + 4, K]` with
//! `coeff: Float[I, J, K]`, written into `out_field: Float[I, J, K]` (the
//! output, `mutated_outputs`; overwritten whole before it is read, so never
//! loaded). With `lap` the 5-point Laplacian of `in_field` on
//! `[I + 2, J + 2]` (`4.0 * x` is exact in Q32), the fluxes `flx`, `fly` are
//! differences of `lap` set to 0 where `res * (difference of in_field) > 0`,
//! and `out = in_field[2:I + 2, 2:J + 2] - coeff * (flx[1:] - flx[:-1] +
//! fly[:, 1:] - fly[:, :-1])`.
//! The test `res * diff > 0` is decided exactly, on the sign of the raw
//! product (NumPy's float64 product has the exact sign): one 132-bit range
//! check (`limit`) instead of a rescale and a 64-bit comparison. The rounded
//! product `floor(res * diff / 2^32)` would be 0 for products in
//! `(0, 2^-32)` and give outputs off by more than the tolerance there. `coeff * (...)` is
//! rounded. Only the `lap` values the fluxes use are computed (its four
//! corners are not), and only the `in_field` elements they and the output
//! read are loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const I: usize = 11; // = out_field_0
const J: usize = 11; // = out_field_1
const K: usize = 10; // = out_field_2
const IN_I: usize = I + 4;
const IN_J: usize = J + 4;

/// `|res * diff| < 2^131` (`|res| < 2^67`, `|diff| < 2^64` for inputs in the
/// window): the comparison gaps below lie in `[0, 2^132)`.
const PRODUCT_BITS: usize = 132;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let in_field = inputs.floats("in_field", &[IN_I, IN_J, K])?;
    let coeff = inputs.floats("coeff", &[I, J, K])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut s = Stencil { q: &q, values: &in_field, input: vec![None; IN_I * IN_J * K], lap: vec![None; (I + 2) * (J + 2) * K] };
    let coeff = q.load_floats(ctx, &coeff);
    let mut flx: Vec<Cell> = Vec::with_capacity((I + 1) * J * K);
    for a in 0..I + 1 {
        for b in 0..J {
            for k in 0..K {
                let res = s.difference(ctx, (a + 1, b + 1, k), (a, b + 1, k));
                let diff = s.input_difference(ctx, (a + 2, b + 2, k), (a + 1, b + 2, k));
                flx.push(limit(&q, ctx, res, diff));
            }
        }
    }
    let mut fly: Vec<Cell> = Vec::with_capacity(I * (J + 1) * K);
    for a in 0..I {
        for b in 0..J + 1 {
            for k in 0..K {
                let res = s.difference(ctx, (a + 1, b + 1, k), (a + 1, b, k));
                let diff = s.input_difference(ctx, (a + 2, b + 2, k), (a + 2, b + 1, k));
                fly.push(limit(&q, ctx, res, diff));
            }
        }
    }
    let mut result: Vec<Cell> = Vec::with_capacity(I * J * K);
    for a in 0..I {
        for b in 0..J {
            for k in 0..K {
                let x = |a: usize, b: usize| (a * J + b) * K + k;
                let y = |a: usize, b: usize| (a * (J + 1) + b) * K + k;
                let gx = q.sub(ctx, flx[x(a + 1, b)], flx[x(a, b)]);
                let gy = q.sub(ctx, fly[y(a, b + 1)], fly[y(a, b)]);
                let divergence = q.add(ctx, gx, gy);
                let scaled = q.mul(ctx, coeff[x(a, b)], divergence);
                let centre = s.input(ctx, (a + 2, b + 2, k));
                result.push(q.sub(ctx, centre, scaled.cell));
            }
        }
    }
    out.floats(&result);
    Ok(())
}

/// `np.where(res * diff > 0, 0, res)`. The outcome is a hinted bit pinned
/// by one range check: `res * diff - 1` (outcome 1) or `-res * diff`
/// (outcome 0) must lie in `[0, 2^132)`, which holds for the true outcome
/// (`|res * diff| < 2^131`) and fails for the other, whose value is
/// negative. (halo2-base's `is_less_than` at 132 bits: 418,860 advice cells
/// for the circuit against 416,220, the same lookup cells.)
fn limit(q: &Q32, ctx: &mut Ctx, res: Cell, diff: Cell) -> Cell {
    let product = q.gate.mul(ctx, res, diff);
    let v = crate::q32::signed(product.value());
    // Hint, pinned by the range check below.
    let positive = ctx.load_witness(Fr::from((v > 0) as u64));
    q.gate.assert_bit(ctx, positive);
    // -product + positive * (2 product - 1) is the gap of the claimed outcome.
    let below = q.gate.neg(ctx, product);
    let swing = q.gate.mul_add(ctx, product, Constant(Fr::from(2)), Constant(-Fr::ONE));
    let gap = q.gate.mul_add(ctx, positive, swing, below);
    q.range.range_check(ctx, gap, PRODUCT_BITS);
    // res if not positive: res - positive * res.
    q.gate.sub_mul(ctx, res, positive, res)
}

/// `in_field` loaded on first use (with its domain check) and `lap`
/// computed on first use.
struct Stencil<'a> {
    q: &'a Q32,
    values: &'a [f64],
    input: Vec<Option<Cell>>,
    lap: Vec<Option<Cell>>,
}

impl Stencil<'_> {
    fn input(&mut self, ctx: &mut Ctx, (i, j, k): (usize, usize, usize)) -> Cell {
        let at = (i * IN_J + j) * K + k;
        *self.input[at].get_or_insert_with(|| self.q.load_float(ctx, self.values[at]))
    }

    fn input_difference(&mut self, ctx: &mut Ctx, p: (usize, usize, usize), r: (usize, usize, usize)) -> Cell {
        let (x, y) = (self.input(ctx, p), self.input(ctx, r));
        self.q.sub(ctx, x, y)
    }

    /// `lap[a, b, k] = 4 in[a + 1, b + 1] - (in[a + 2, b + 1] + in[a, b + 1]
    /// + in[a + 1, b + 2] + in[a + 1, b])`.
    fn lap(&mut self, ctx: &mut Ctx, (a, b, k): (usize, usize, usize)) -> Cell {
        let at = (a * (J + 2) + b) * K + k;
        if let Some(cell) = self.lap[at] {
            return cell;
        }
        let points = [(a + 1, b + 1), (a + 2, b + 1), (a, b + 1), (a + 1, b + 2), (a + 1, b)];
        let cells: Vec<Cell> = points.into_iter().map(|(i, j)| self.input(ctx, (i, j, k))).collect();
        let weights = [Fr::from(4), -Fr::ONE, -Fr::ONE, -Fr::ONE, -Fr::ONE];
        let cell = self.q.gate.inner_product(ctx, cells.into_iter().map(Existing), weights.into_iter().map(Constant));
        self.lap[at] = Some(cell);
        cell
    }

    fn difference(&mut self, ctx: &mut Ctx, p: (usize, usize, usize), r: (usize, usize, usize)) -> Cell {
        let (x, y) = (self.lap(ctx, p), self.lap(ctx, r));
        self.q.sub(ctx, x, y)
    }
}
