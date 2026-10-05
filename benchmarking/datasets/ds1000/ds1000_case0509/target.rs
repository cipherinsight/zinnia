//! DS-1000 case 0509, halo2-base baseline.
//! Kernel: on `im: Float[10, 10]`, `mask = im == 0`; crop `im` to the rows and
//! columns spanned by those containing a 0
//! (`im[rows.min():rows.max()+1, cols.min():cols.max()+1]`), or `np.array([])`
//! when no entry is 0.
//! Output: public (rows, cols), then the crop compact and row-major,
//! zero-padded to 10 x 10; the empty result is (0, 0).
//! Construction (each test made once, `==` exact as in the kernel):
//! - a row contains a 0 iff the product of its entries is 0 (exact for any
//!   field elements: a field has no zero divisors), one zero test per row and
//!   per column;
//! - lead_i = "rows 0..=i contain no 0" and after_i = "rows i.. contain no 0"
//!   are running products of the complements of those bits; the first row with
//!   a 0 is the one-hot top_d = lead_{d-1} - lead_d, and the height is
//!   10 - sum(lead) - sum(after) + 10 lead_9 (0 when no entry is 0); the same
//!   for columns;
//! - out[u][v] = im[top + u][left + v] is a one-hot shift of the rows, then of
//!   the columns. Rows and columns outside the span need not be zero here, so
//!   the selector of output row u reading source row d + u is
//!   top_d * (1 - after_{d+u}), which is 0 past the span (likewise for columns);
//! - the crop, now at the top-left of a zero 10 x 10 grid, is laid out with
//!   stride `width`: one fixed reshuffle per possible width, selected by the
//!   one-hot of `width`.

use crate::prelude::*;

const R: usize = 10;
const C: usize = 10;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let im = inputs.floats("im", &[R, C])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let im = q.load_floats(ctx, &im);
    let row_hit: Vec<Cell> = (0..R).map(|i| has_zero(&q, ctx, (0..C).map(|j| im[i * C + j]))).collect();
    let col_hit: Vec<Cell> = (0..C).map(|j| has_zero(&q, ctx, (0..R).map(|i| im[i * C + j]))).collect();
    let (top, row_after, height) = extent(&q, ctx, &row_hit);
    let (left, col_after, width) = extent(&q, ctx, &col_hit);
    let row_sel = span_selectors(&q, ctx, &top, &row_after);
    let col_sel = span_selectors(&q, ctx, &left, &col_after);
    let grid = shift(&q, ctx, &im, |u, d| row_sel[u][d], |v, e| col_sel[v][e]);
    let crop = compact(&q, ctx, &grid, width);
    out.dynamic2(height, width, &crop, Kind::Float, R, C);
    Ok(())
}

/// 1 if some value is 0: their product is 0.
fn has_zero(q: &Q32, ctx: &mut Ctx, mut xs: impl Iterator<Item = Cell>) -> Cell {
    let first = xs.next().expect("non-empty");
    let prod = xs.fold(first, |p, x| q.gate.mul(ctx, p, x));
    q.gate.is_zero(ctx, prod)
}

/// From `hit[i]` = 1 when line i is selected: the one-hot of the first
/// selected line, `after[i]` = 1 when line i comes after the last selected
/// one, and the number of lines from the first to the last selected one (0
/// when none is selected).
fn extent(q: &Q32, ctx: &mut Ctx, hit: &[Cell]) -> (Vec<Cell>, Vec<Cell>, Cell) {
    let n = hit.len();
    let mut lead = vec![q.gate.not(ctx, hit[0])]; // lead[i]: no line in 0..=i selected
    for i in 1..n {
        lead.push(q.gate.mul_not(ctx, hit[i], lead[i - 1]));
    }
    let mut after = vec![q.gate.not(ctx, hit[n - 1]); n]; // after[i]: no line in i.. selected
    for i in (0..n - 1).rev() {
        after[i] = q.gate.mul_not(ctx, hit[i], after[i + 1]);
    }
    let mut first = vec![hit[0]];
    for d in 1..n {
        first.push(q.sub(ctx, lead[d - 1], lead[d]));
    }
    // n - sum(lead) - sum(after) + n * lead[n-1]
    let terms = std::iter::once(Constant(fe(n as i128)))
        .chain(lead.iter().chain(&after).map(|c| Existing(*c)))
        .chain(std::iter::once(Existing(lead[n - 1])));
    let coefs = std::iter::once(Constant(Fr::ONE))
        .chain(std::iter::repeat(Constant(-Fr::ONE)).take(2 * n))
        .chain(std::iter::once(Constant(fe(n as i128))));
    let len = q.gate.inner_product(ctx, terms, coefs);
    (first, after, len)
}

/// sel[u][d] = first[d] * (1 - after[d + u]): output line u reads source line
/// d + u only while that line is inside the span. For u = 0 it is first[d]
/// (the first selected line is inside the span).
fn span_selectors(q: &Q32, ctx: &mut Ctx, first: &[Cell], after: &[Cell]) -> Vec<Vec<Cell>> {
    let n = first.len();
    let mut sel = vec![first.to_vec()];
    for u in 1..n {
        sel.push((0..n - u).map(|d| q.gate.mul_not(ctx, after[d + u], first[d])).collect());
    }
    sel
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
