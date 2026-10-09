//! Source Deriche recurrences and coefficient evaluation, canonical Q32.
//! exp(-2*alpha) is evaluated directly, not the square of exp(-alpha).
//! 2**(-alpha) follows canonical exp((-alpha)*log(2)), where the canonical
//! Q32 log(2) constant is 2977044472. Repeated identical scalar expressions
//! and zero products are reused; all nontrivial product rounding remains.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const R: usize = 16; // = imgIn_0
const C: usize = 8; // = imgIn_1

/// The coefficients of one recurrence pair: `y1[j] = a1 x[j] + a2 x[j - 1]
/// + b1 y1[j - 1] + b2 y1[j - 2]` and `y2[j] = a3 x[j + 1] + a4 x[j + 2]
/// + b1 y2[j + 1] + b2 y2[j + 2]`.
struct Coefficients {
    a1: Cell,
    a2: Cell,
    a3: Cell,
    a4: Cell,
    b1: Cell,
    b2: Cell,
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let img = inputs.floats("imgIn", &[R, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let img = q.load_floats(ctx, &img);
    let one = Constant(fe(SCALE));
    let minus_alpha = q.neg(ctx, alpha);
    let minus_alpha = q.bound(ctx, minus_alpha);
    let e1 = q.canonical_exp(ctx, minus_alpha).cell; // np.exp(-alpha)
    let two_alpha = q.add(ctx, alpha, alpha);
    let two_alpha = q.bound(ctx, two_alpha);
    let e_two = q.canonical_exp(ctx, two_alpha).cell; // np.exp(2.0 * alpha)
    let minus_two_alpha = q.neg(ctx, two_alpha.cell);
    let minus_two_alpha = q.bound(ctx, minus_two_alpha);
    let e2 = q.canonical_exp(ctx, minus_two_alpha).cell; // np.exp(-2.0 * alpha)
    // k = (1.0 - e1) * (1.0 - e1) / (1.0 + alpha * e1 - e_two)
    let one_minus_e1 = q.sub(ctx, one, e1);
    let numerator = q.mul(ctx, one_minus_e1, one_minus_e1);
    let alpha_e1 = q.mul(ctx, alpha, e1);
    let denominator = q.add(ctx, one, alpha_e1.cell);
    let denominator = q.sub(ctx, denominator, e_two);
    let denominator = q.bound(ctx, denominator);
    let k = q.div(ctx, numerator.cell, denominator).cell;
    // a2 = k * e1 * (alpha - 1.0), a3 = k * e1 * (alpha + 1.0), a4 = -k * e2
    let k_e1 = q.mul(ctx, k, e1).cell;
    let alpha_minus_one = q.sub(ctx, alpha, one);
    let alpha_plus_one = q.add(ctx, alpha, one);
    let minus_k = q.neg(ctx, k);
    // b1 = 2.0 ** (-alpha) = exp(-alpha * ln 2), b2 = -e2
    let exponent = q.mul(ctx, minus_alpha.cell, Constant(fe(quantize(std::f64::consts::LN_2))));
    let coefficients = Coefficients {
        a1: k,
        a2: q.mul(ctx, k_e1, alpha_minus_one).cell,
        a3: q.mul(ctx, k_e1, alpha_plus_one).cell,
        a4: q.mul(ctx, minus_k, e2).cell,
        b1: q.canonical_exp(ctx, exponent).cell,
        b2: q.neg(ctx, e2),
    };

    // Along the rows, then along the columns of the result.
    let mut horizontal = vec![img[0]; R * C];
    for i in 0..R {
        let row = filter(&q, ctx, &img[i * C..(i + 1) * C], &coefficients);
        horizontal[i * C..(i + 1) * C].copy_from_slice(&row);
    }
    let mut result = vec![img[0]; R * C];
    for j in 0..C {
        let column: Vec<Cell> = (0..R).map(|i| horizontal[i * C + j]).collect();
        for (i, v) in filter(&q, ctx, &column, &coefficients).into_iter().enumerate() {
            result[i * C + j] = v;
        }
    }
    out.floats(&result);
    Ok(())
}

/// `y1 + y2` along `x` (`c1 * (y1 + y2)` with `c1 = 1`).
fn filter(q: &Q32, ctx: &mut Ctx, x: &[Cell], c: &Coefficients) -> Vec<Cell> {
    let n = x.len();
    let product = |ctx: &mut Ctx, a: Cell, b: Cell| q.mul(ctx, a, b).cell;
    // y1[0] = a1 x[0]; y1[1] = a1 x[1] + a2 x[0] + b1 y1[0]; then four terms.
    let mut y1: Vec<Cell> = Vec::with_capacity(n);
    for j in 0..n {
        let mut terms = vec![product(ctx, c.a1, x[j])];
        if j >= 1 {
            terms.push(product(ctx, c.a2, x[j - 1]));
            terms.push(product(ctx, c.b1, y1[j - 1]));
        }
        if j >= 2 {
            terms.push(product(ctx, c.b2, y1[j - 2]));
        }
        y1.push(q.sum(ctx, terms));
    }
    // y2[n - 1] = 0 (None); y2[n - 2] = a3 x[n - 1]; then the terms that are
    // not products with y2[n - 1].
    let mut y2: Vec<Option<Cell>> = vec![None; n];
    for j in (0..n - 1).rev() {
        let mut terms = vec![product(ctx, c.a3, x[j + 1])];
        if j + 2 < n {
            terms.push(product(ctx, c.a4, x[j + 2]));
            terms.push(product(ctx, c.b1, y2[j + 1].expect("set")));
        }
        if j + 3 < n {
            terms.push(product(ctx, c.b2, y2[j + 2].expect("set")));
        }
        y2[j] = Some(q.sum(ctx, terms));
    }
    y1.iter().zip(&y2).map(|(a, b)| b.map_or(*a, |b| q.add(ctx, *a, b))).collect()
}
