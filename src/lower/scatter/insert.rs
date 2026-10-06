//! `np.insert(a, pos, values, axis)` at a runtime position: one slice
//! inserted along an axis of length `n`.
//!
//! Terms. The axis splits the source into `n` *rows* (slices along the
//! axis), each of `L` *lanes* (the cells of a slice). `e` is the inserted
//! slice (`L` lanes, already in the output dtype). The output has `n + 1`
//! rows. With `P` NumPy's insertion point (`pos`, or `pos + n` for a
//! negative `pos`), row `k` of the output is `a[k]` for `k < P`, `e` for
//! `k = P` and `a[k - 1]` for `k > P`.
//!
//! With the pinned one-hot `h[k] = [P == k]` and thermometer
//! `g[k] = [P < k]` of [`super::position`] (`g[0] = 0`, `g[n + 1] = 1`),
//! every output row is one expression in those bits, the same for every
//! position, shared by all `L` lanes:
//!
//! ```text
//! out[k] = (1 - g[k+1]) a[k] + g[k] a[k-1] + h[k] e
//! ```
//!
//! (exactly one of the three bits is 1: `k < P`, `k > P` or `k = P`). Terms
//! with `a[-1]` or `a[n]` have a zero coefficient and are dropped. There is
//! no comparison and no select. Two forms compute it:
//!
//! * [`Form::Complement`]: `(1 - g[k+1])` is `r[k+1] = [P >= k+1]`, a second
//!   running sum (from the end), so every term is a product of a bit and a
//!   cell: `out[k] = r[k+1] a[k] + g[k] a[k-1] + h[k] e`.
//!   halo2-base: 10 cells per output cell (a product gate and two chained
//!   multiply-adds) plus 3 per row for `r`. Noir: three products per cell.
//! * [`Form::SharedProduct`]: `t[k] = g[k+1] a[k]` is shared by rows `k` and
//!   `k + 1`: `out[k] = a[k] - t[k] + t[k-1] + h[k] e`. Noir: two products
//!   per cell. halo2-base: 15 cells per output cell when `t[k]` is a cell,
//!   11 when the backend fuses each use of `t[k]` into its sum.
//!
//! **Preconditions**: `n` and `L` are compile-time constants; `pos` is an
//! integer; `a` and `e` are cells of one dtype (int64 or Q32: the products
//! with a bit need no rescale).
//! **Postconditions**: the `n + 1` output rows, equal to NumPy's result; the
//! fact `-n <= pos <= n` (no witness otherwise, as NumPy raises).
//! **Soundness and completeness**: the bits are pinned by
//! [`super::position::position_one_hot`]; given them, every output cell is a
//! fixed polynomial of the inputs (no hint), whose value is the row the
//! case analysis above selects.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::arith::{bit_times, difference, sum};
use super::position::{position_one_hot, Positions};

/// How the output rows are computed (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Complement,
    SharedProduct,
}

/// The `n + 1` output rows of inserting `inserted` before row `pos` of
/// `rows` (`rows.len() = n`, every row and `inserted` of the same length).
pub fn insert_row(
    b: &mut IRBuilder,
    rows: &[Vec<Value>],
    inserted: &[Value],
    pos: &Value,
    dtype: NumberType,
    form: Form,
) -> Vec<Vec<Value>> {
    let n = rows.len();
    let point = position_one_hot(b, pos, n, Positions::InsertionPoint);
    let (h, g) = (&point.hot, &point.below);
    let lanes = inserted.len();
    match form {
        Form::Complement => {
            let r = point.at_or_above(b);
            (0..=n)
                .map(|k| {
                    (0..lanes)
                        .map(|j| {
                            let mut terms = Vec::with_capacity(3);
                            if k < n {
                                terms.push(bit_times(b, &r[k + 1], &rows[k][j], dtype));
                            }
                            if k > 0 {
                                terms.push(bit_times(b, &g[k], &rows[k - 1][j], dtype));
                            }
                            terms.push(bit_times(b, &h[k], &inserted[j], dtype));
                            sum(b, &terms, dtype)
                        })
                        .collect()
                })
                .collect()
        }
        Form::SharedProduct => {
            // t[k][j] = g[k+1] a[k][j] for k < n.
            let t: Vec<Vec<Value>> = (0..n)
                .map(|k| (0..lanes).map(|j| bit_times(b, &g[k + 1], &rows[k][j], dtype)).collect())
                .collect();
            (0..=n)
                .map(|k| {
                    (0..lanes)
                        .map(|j| {
                            // (a[k] + t[k-1]) + h[k] e - t[k], in the order
                            // of a multiply-add chain.
                            let mut terms = Vec::with_capacity(3);
                            if k < n {
                                terms.push(rows[k][j].clone());
                            }
                            if k > 0 {
                                terms.push(t[k - 1][j].clone());
                            }
                            terms.push(bit_times(b, &h[k], &inserted[j], dtype));
                            let out = sum(b, &terms, dtype);
                            if k < n {
                                difference(b, &out, &t[k][j], dtype)
                            } else {
                                out
                            }
                        })
                        .collect()
                })
                .collect()
        }
    }
}
