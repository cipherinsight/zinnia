//! DS-1000 case 0510, halo2-base baseline.
//! Kernel: on `im: Float[10, 10]`, `mask = im == 0`; crop `im` to the bounding
//! box of its nonzero entries (`rows`, `cols` from `~mask`;
//! `im[rows.min():rows.max()+1, cols.min():cols.max()+1]`), or `np.array([])`
//! when every entry is 0.
//! Output: public (rows, cols), then the crop compact and row-major,
//! zero-padded to 10 x 10; the empty result is (0, 0).
//! Construction (each test made once, `==` exact as in the kernel):
//! - one zero test per row and per column, exact for any field elements (the
//!   inputs carry no range proof): out = 1 - sum_j x_j w_j with x_j * out = 0
//!   for every j, w a hint;
//! - lead_i = "rows 0..=i are all zero" and trail_i = "rows i.. are all zero"
//!   are running products of those bits; the first nonzero row is the one-hot
//!   top_d = lead_{d-1} - lead_d, and the height is
//!   10 - sum(lead) - sum(trail) + 10 lead_9 (0 for an all-zero image); the same
//!   for columns;
//! - out[u][v] = im[top + u][left + v] is a one-hot shift of the rows, then of
//!   the columns. Positions past the box read rows or columns outside the
//!   bounding box, which are all zero, so the padding needs no mask;
//! - the crop, now at the top-left of a zero grid, is laid out with stride
//!   `width`: one fixed reshuffle per possible width, selected by the one-hot
//!   of `width`.

use crate::prelude::*;

const R: usize = 10;
const C: usize = 10;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let im = inputs.floats("im", &[R, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let im = q.load_floats(ctx, &im);
    let row_zero: Vec<Cell> = (0..R).map(|i| all_zero(&q, ctx, (0..C).map(|j| im[i * C + j]))).collect();
    let col_zero: Vec<Cell> = (0..C).map(|j| all_zero(&q, ctx, (0..R).map(|i| im[i * C + j]))).collect();
    let (top, height) = extent(&q, ctx, &row_zero);
    let (left, width) = extent(&q, ctx, &col_zero);
    let grid = shift(&q, ctx, &im, |_, d| top[d], |_, e| left[e]);
    let crop = compact(&q, ctx, &grid, width);
    out.dynamic2(height, width, &crop, Kind::Float, R, C);
    Ok(())
}

/// 1 if every value is 0, else 0, for any field elements:
/// out = 1 - sum_j x_j w_j and x_j * out = 0 for every j. A nonzero x_k forces
/// out = 0 (w = minus the inverse of the first nonzero x, zeros elsewhere,
/// gives the sum 1); all-zero xs force out = 1. The hints w_j are not unique,
/// but out is, and nothing else reads them.
fn all_zero(q: &Q32, ctx: &mut Ctx, xs: impl Iterator<Item = Cell>) -> Cell {
    let xs: Vec<Cell> = xs.collect();
    let first = xs.iter().position(|x| *x.value() != Fr::ZERO);
    let w = (0..xs.len()).map(|j| match first {
        Some(k) if k == j => Witness(-xs[k].value().invert().unwrap()),
        _ => Witness(Fr::ZERO),
    });
    let one = std::iter::once(Constant(Fr::ONE));
    let out = q.gate.inner_product(ctx, one.clone().chain(xs.iter().map(|x| Existing(*x))), one.chain(w));
    // x_j * out = 0: one gate a + b * c = d per j, rows (0, x_j, out, 0), the
    // zeros shared by neighbouring gates.
    let mut cells = vec![Constant(Fr::ZERO)];
    for x in &xs {
        cells.extend([Existing(*x), Existing(out), Constant(Fr::ZERO)]);
    }
    ctx.assign_region(cells, (0..xs.len()).map(|j| 3 * j as isize));
    out
}

/// From `skip[i]` = 1 when line i is all zero: the one-hot of the first line
/// that is not, and the number of lines from it to the last one that is not
/// (0 when every line is skipped).
fn extent(q: &Q32, ctx: &mut Ctx, skip: &[Cell]) -> (Vec<Cell>, Cell) {
    let n = skip.len();
    let mut lead = vec![skip[0]]; // lead[i]: lines 0..=i skipped
    for i in 1..n {
        lead.push(q.gate.mul(ctx, lead[i - 1], skip[i]));
    }
    let mut trail = vec![skip[n - 1]]; // trail[t]: lines n-1-t.. skipped
    for i in (0..n - 1).rev() {
        trail.push(q.gate.mul(ctx, trail[trail.len() - 1], skip[i]));
    }
    let mut first = vec![q.gate.not(ctx, lead[0])];
    for d in 1..n {
        first.push(q.sub(ctx, lead[d - 1], lead[d]));
    }
    // n - sum(lead) - sum(trail) + n * lead[n-1]
    let terms = std::iter::once(Constant(fe(n as i128)))
        .chain(lead.iter().chain(&trail).map(|c| Existing(*c)))
        .chain(std::iter::once(Existing(lead[n - 1])));
    let coefs = std::iter::once(Constant(Fr::ONE))
        .chain(std::iter::repeat(Constant(-Fr::ONE)).take(2 * n))
        .chain(std::iter::once(Constant(fe(n as i128))));
    let len = q.gate.inner_product(ctx, terms, coefs);
    (first, len)
}

/// out[u][v] = sum over d, e of row_sel(u, d) * col_sel(v, e) * src[u + d][v + e]
/// (positions past the end of `src` contribute nothing), as a shift of the
/// rows followed by a shift of the columns.
fn shift(
    q: &Q32,
    ctx: &mut Ctx,
    src: &[Cell],
    row_sel: impl Fn(usize, usize) -> Cell,
    col_sel: impl Fn(usize, usize) -> Cell,
) -> Vec<Cell> {
    let mut rows = Vec::with_capacity(R * C);
    for u in 0..R {
        for j in 0..C {
            let sel = (0..R - u).map(|d| Existing(row_sel(u, d)));
            let vals = (0..R - u).map(|d| Existing(src[(u + d) * C + j]));
            rows.push(q.gate.inner_product(ctx, sel, vals));
        }
    }
    let mut out = Vec::with_capacity(R * C);
    for u in 0..R {
        for v in 0..C {
            let sel = (0..C - v).map(|e| Existing(col_sel(v, e)));
            let vals = (0..C - v).map(|e| Existing(rows[u * C + v + e]));
            out.push(q.gate.inner_product(ctx, sel, vals));
        }
    }
    out
}

/// The crop held at the top-left of the R x C `grid` (zero elsewhere), laid out
/// compact and row-major for its data-dependent width w: slot s holds
/// grid[s / w][s % w]. One fixed reshuffle per possible width, selected by the
/// one-hot [w == W] (w = 0 leaves the all-zero grid, so every slot is 0).
fn compact(q: &Q32, ctx: &mut Ctx, grid: &[Cell], width: Cell) -> Vec<Cell> {
    let is_width: Vec<Cell> = (1..=C).map(|w| q.eq(ctx, width, Constant(fe(w as i128)))).collect();
    (0..R * C)
        .map(|s| {
            let widths: Vec<usize> = (1..=C).filter(|w| s / w < R).collect();
            let sel = widths.iter().map(|w| Existing(is_width[w - 1]));
            let vals = widths.iter().map(|w| Existing(grid[(s / w) * C + s % w]));
            q.gate.inner_product(ctx, sel, vals)
        })
        .collect()
}
