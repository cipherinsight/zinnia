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
//! no approximation error). `np.float32(beta)` is computed in Q32 like any
//! float (NPB_BASELINES.md section 5.4): rounding `beta` to float32 changes
//! the outcome only when `beta * dE` is within 6e-8 relative of ln 2; the
//! fixture's values are far from it (`beta * dE` is 0 or 3.2,
//! against ln 2 = 0.69). `dE` is an int64 (a comparison puts it in the window); the
//! product `beta * dE` is compared at 127 bits, so any int64 `dE` and Q32
//! `beta` have a proof, as NumPy's float64 product has a value.

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
    let beta = q.load_float(ctx, beta);
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
