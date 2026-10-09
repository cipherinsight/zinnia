//! pythran_ising__ising_step (Pythran ising), halo2-base baseline.
//! Kernel: one checkerboard sweep of the Ising model on `field: Int[N, M]`
//! (returned): the sites `(n, m)` in the kernel's order (parity of `n`, then
//! of `m`), each updated in turn from the current field. A site sums its
//! eight periodic neighbours (`clamp`; the indices are loop constants),
//! `dE = 2 * field[n, m] * total`, and flips (`field[n, m] *= -1`) when
//! `dE <= 0` or `np.exp(-dE * beta) > 0.5`.
//! `exp` is increasing, so `exp(-dE * beta) > 0.5` is decided exactly as
//! `beta * dE < -ln(0.5)` on the Q32 code of `beta * dE` (an integer
//! comparison with the constant `-round(ln(0.5) * 2^32)`; no `exp` gadget,
//! no approximation error). beta is clamped to [0,1] before rounding to
//! float32: for positive integer dE, beta <= 0 always flips and beta >= 1
//! never flips, so clamping preserves the decision. Rounding the clamped
//! Q32 code to 24 significant bits implements np.float32, ties to even.
//! The product beta*dE is compared at 127 bits.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = field_0
const M: usize = 2; // = field_1

/// `-round(ln(0.5) * 2^32)`: for an integer code `c`, `c / 2^32 < -ln(0.5)`
/// exactly when `c < NEG_LN_HALF` (`ln(0.5) * 2^32` is not an integer).
const NEG_LN_HALF: i128 = 2977044472;
/// `|beta * dE| < 2^126` (both codes in the window): its offset by `2^126`
/// lies in `[0, 2^127)`, as `is_less_than` needs.
const PRODUCT_BITS: usize = 127;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let field = inputs.ints("field", &[N, M])?;
    let beta = inputs.float("beta")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut field = q.load_ints(ctx, &field);
    let beta = q.load_float_bounded(ctx, beta);
    let beta = acceptance_beta(&q, ctx, beta);
    for n_offset in 0..2 {
        for m_offset in 0..2 {
            for n in (n_offset..N).step_by(2) {
                for m in (m_offset..M).step_by(2) {
                    update(&q, ctx, &mut field, beta, n, m);
                }
            }
        }
    }
    out.ints(&field);
    Ok(())
}

/// The kernel's `clamp(v, n)`: a periodic index.
fn clamp(v: isize, n: usize) -> usize {
    v.rem_euclid(n as isize) as usize
}

fn update(q: &Q32, ctx: &mut Ctx, field: &mut [Cell], beta: Cell, n: usize, m: usize) {
    let mut neighbours: Vec<Cell> = Vec::with_capacity(8);
    for i in n as isize - 1..=n as isize + 1 {
        for j in m as isize - 1..=m as isize + 1 {
            if (i, j) != (n as isize, m as isize) {
                neighbours.push(field[clamp(i, N) * M + clamp(j, M)]);
            }
        }
    }
    let total = q.sum(ctx, neighbours);
    let site = field[n * M + m];
    let half = q.gate.mul(ctx, site, total);
    let de = q.gate.mul(ctx, half, Constant(Fr::from(2)));
    let de = q.bound(ctx, de);
    let non_positive = q.lt_const(ctx, de, 1);
    // exp(-dE * beta) > 0.5, i.e. beta * dE < -ln(0.5).
    let product = q.gate.mul(ctx, beta, de.cell);
    let shift = 1i128 << (PRODUCT_BITS - 1);
    let shifted = q.gate.add(ctx, product, Constant(fe(shift)));
    let exp_test = q.range.is_less_than(ctx, shifted, Constant(fe(shift + NEG_LN_HALF)), PRODUCT_BITS);
    let flip = q.gate.or(ctx, non_positive, exp_test);
    // field[n, m] *= -1 when flipping: site - 2 * flip * site.
    let twice = q.gate.mul(ctx, flip, Constant(Fr::from(2)));
    field[n * M + m] = q.gate.sub_mul(ctx, site, twice, site);
}

/// Equivalent beta for the acceptance predicate, with exact float32 rounding
/// of the Q32 input. Only shifts 0..9 occur after clamping to [0,1].
fn acceptance_beta(q: &Q32, ctx: &mut Ctx, beta: I64) -> Cell {
    let zero = q.float_const(ctx, 0.0);
    let zero = q.bound(ctx, zero);
    let one = q.float_const(ctx, 1.0);
    let one = q.bound(ctx, one);
    let nonnegative = q.max(ctx, beta, zero);
    let clamped = q.min(ctx, nonnegative, one);
    let code = clamped.value() as u128;
    let shift = (128 - code.leading_zeros()).saturating_sub(24);
    // Hint pinned by a one-hot shift and its magnitude interval.
    let shift = ctx.load_witness(Fr::from(shift as u64));
    round_float32(q, ctx, clamped.cell, shift)
}

fn round_float32(q: &Q32, ctx: &mut Ctx, code: Cell, shift: Cell) -> Cell {
    let mut choices = Vec::new();
    let mut factors = Vec::new();
    let mut lows = Vec::new();
    let mut highs = Vec::new();
    for s in 0..10 {
        let choice = q.gate.is_equal(ctx, shift, Constant(Fr::from(s)));
        choices.push(choice);
        factors.push(q.gate.mul(ctx, choice, Constant(fe(1 << s))));
        lows.push(q.gate.mul(ctx, choice, Constant(fe(if s == 0 { 0 } else { 1 << (23 + s) }))));
        highs.push(q.gate.mul(ctx, choice, Constant(fe(1 << (24 + s)))));
    }
    let count = q.sum(ctx, choices);
    q.gate.assert_is_const(ctx, &count, &Fr::ONE);
    let factor = q.sum(ctx, factors);
    let low = q.sum(ctx, lows);
    let high = q.sum(ctx, highs);
    let low_gap = q.sub(ctx, code, low);
    let high_gap = q.sub(ctx, high, code);
    let high_gap = q.sub(ctx, high_gap, Constant(Fr::ONE));
    q.range.range_check(ctx, low_gap, 32);
    q.range.range_check(ctx, high_gap, 32);
    let factor_bounded = q.bound(ctx, factor);
    let (mantissa, rem) = q.floor_div_pos(ctx, code, factor_bounded);
    let (_, odd) = q.divmod_small(ctx, mantissa.cell, 2, 24);
    let twice_rem = q.gate.mul(ctx, rem, Constant(Fr::from(2)));
    let above = q.range.is_less_than(ctx, factor, twice_rem, 10);
    let tie = q.eq(ctx, twice_rem, factor);
    let tie_up = q.gate.mul(ctx, tie, odd);
    let up = q.gate.or(ctx, above, tie_up);
    let rounded = q.add(ctx, mantissa.cell, up);
    q.gate.mul(ctx, rounded, factor)
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
    fn float32_rounding_and_clamping() {
        assert!(accepts(|q, ctx| {
            for (code, expected) in [(186065279, 186065280), (-1, 0),
                (2*SCALE, SCALE), (16777217, 16777216), (16777219, 16777220)] {
                let x = ctx.load_witness(fe(code));
                let x = q.bound(ctx, x);
                let rounded = acceptance_beta(q, ctx, x);
                q.gate.assert_is_const(ctx, &rounded, &fe(expected));
            }
        }));
    }

    #[test]
    fn float32_rejects_wrong_or_nonintegral_exponent() {
        for shift in [fe(0), fe(5), fe(1) * fe(2).invert().unwrap()] {
            assert!(!accepts(|q, ctx| {
                let x = ctx.load_witness(fe(186065279));
                let shift = ctx.load_witness(shift);
                round_float32(q, ctx, x, shift);
            }));
        }
    }
}
