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

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::arith::{bit_times, difference, sum};
use super::position::{position_one_hot, PositionOneHot, Positions};

/// How the output rows are computed (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Complement,
    SharedProduct,
}

impl Form {
    /// Cost of the output rows on `target` beyond the position one-hot, for
    /// `n + 1` output rows of `lanes` cells (calibrated on 0363/0364:
    /// halo2-base 11 cells per cell shared, 10 per cell plus 3 per row
    /// complement; Noir about 2 gates per cell either way, plus 4/3 per row
    /// for the complement's second running sum).
    pub fn cost(self, target: crate::lower::Target, n: usize, lanes: usize) -> u64 {
        let (rows, cells) = ((n + 1) as u64, ((n + 1) * lanes) as u64);
        match (target, self) {
            (crate::lower::Target::Halo2Base, Form::SharedProduct) => 11 * cells,
            (crate::lower::Target::Halo2Base, Form::Complement) => 10 * cells + 3 * rows,
            (crate::lower::Target::Noir, Form::SharedProduct) => 2 * cells,
            (crate::lower::Target::Noir, Form::Complement) => 2 * cells + (4 * rows).div_ceil(3),
        }
    }
}

/// Cost of the pinned position one-hot of an insertion into `n` rows
/// (`n + 1` positions; see `super::position`): halo2-base 11 cells per
/// position and about 15 once, Noir about 2 gates per position and 40 once.
pub fn position_cost(target: crate::lower::Target, n: usize) -> u64 {
    let positions = (n + 1) as u64;
    match target {
        crate::lower::Target::Halo2Base => 11 * positions + 15,
        crate::lower::Target::Noir => 2 * positions + 40,
    }
}

/// An insertion emitted cell by cell: the position one-hot when it starts,
/// each output cell (and the products it shares) on demand.
pub struct Insertion {
    rows: Vec<Vec<Value>>,
    inserted: Vec<Value>,
    dtype: NumberType,
    form: Form,
    point: PositionOneHot,
    /// `r[k] = [P >= k]` (the complement form).
    at_or_above: Vec<Value>,
    /// `t[k][j] = g[k+1] a[k][j]` by `(k, j)` (the shared-product form).
    shifted: HashMap<(usize, usize), Value>,
}

impl Insertion {
    /// Pins NumPy's insertion point of `pos` into `rows.len()` rows.
    pub fn new(
        b: &mut IRBuilder,
        rows: Vec<Vec<Value>>,
        inserted: Vec<Value>,
        pos: &Value,
        dtype: NumberType,
        form: Form,
    ) -> Insertion {
        let point = position_one_hot(b, pos, rows.len(), Positions::InsertionPoint);
        let at_or_above = match form {
            Form::Complement => point.at_or_above(b),
            Form::SharedProduct => Vec::new(),
        };
        Insertion { rows, inserted, dtype, form, point, at_or_above, shifted: HashMap::new() }
    }

    /// Cell `j` of output row `k` (`k` in `0..=n`).
    pub fn cell(&mut self, b: &mut IRBuilder, k: usize, j: usize) -> Value {
        let n = self.rows.len();
        let (h, g, dtype) = (&self.point.hot, &self.point.below, self.dtype);
        match self.form {
            Form::Complement => {
                let mut terms = Vec::with_capacity(3);
                if k < n {
                    terms.push(bit_times(b, &self.at_or_above[k + 1], &self.rows[k][j], dtype));
                }
                if k > 0 {
                    terms.push(bit_times(b, &g[k], &self.rows[k - 1][j], dtype));
                }
                terms.push(bit_times(b, &h[k], &self.inserted[j], dtype));
                sum(b, &terms, dtype)
            }
            Form::SharedProduct => {
                // (a[k] + t[k-1]) + h[k] e - t[k], in the order of a
                // multiply-add chain.
                let mut terms = Vec::with_capacity(3);
                if k < n {
                    terms.push(self.rows[k][j].clone());
                }
                if k > 0 {
                    terms.push(self.shifted(b, k - 1, j));
                }
                terms.push(bit_times(b, &self.point.hot[k], &self.inserted[j], dtype));
                let out = sum(b, &terms, dtype);
                if k < n {
                    let t = self.shifted(b, k, j);
                    difference(b, &out, &t, dtype)
                } else {
                    out
                }
            }
        }
    }

    /// `t[k][j] = g[k+1] a[k][j]`, emitted once.
    fn shifted(&mut self, b: &mut IRBuilder, k: usize, j: usize) -> Value {
        if let Some(t) = self.shifted.get(&(k, j)) {
            return t.clone();
        }
        let t = bit_times(b, &self.point.below[k + 1], &self.rows[k][j], self.dtype);
        self.shifted.insert((k, j), t.clone());
        t
    }
}
