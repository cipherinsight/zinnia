//! Source sequential Ising sweep with the literal exp(-dE*beta)>0.5
//! predicate. np.float32 intentionally uses the shared Q32 float format.
//! No log-threshold rewrite or binary32 quantization outside that format.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = field_0
const M: usize = 2; // = field_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let field = inputs.ints("field", &[N, M])?;
    let beta = inputs.float("beta")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut field = q.load_ints(ctx, &field);
    let beta = q.load_float_bounded(ctx, beta);
    let beta = beta.cell;
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
    // Frozen IR retains the original exp operand even in the inactive branch.
    let negative_de = q.neg(ctx, de.cell);
    let argument = q.gate.mul(ctx, negative_de, beta);
    let argument = q.bound(ctx, argument);
    let exponential = q.canonical_exp(ctx, argument);
    let exp_test = q.gt_const(ctx, exponential, SCALE / 2);
    let flip = q.gate.or(ctx, non_positive, exp_test);
    // field[n, m] *= -1 when flipping: site - 2 * flip * site.
    let twice = q.gate.mul(ctx, flip, Constant(Fr::from(2)));
    field[n * M + m] = q.gate.sub_mul(ctx, site, twice, site);
}
