//! Window gather: `x[s0:e0, s1:e1, ...]` with runtime bounds as a shift of
//! the whole array by a shared one-hot per axis.
//!
//! The element-wise lowering read every output cell on its own: per output
//! position a clamp, a box test, an address and a gather by indicators of
//! the address, and a padding select (about 75 halo2-base cells per output
//! of a 10x10 crop). Here the index map of each axis is affine with one
//! runtime offset, `u -> start + u`, so one one-hot of `start` serves every
//! output position of that axis:
//!
//! ```text
//! y[.., u, ..] = sum_d sel[u][d] * x[.., u + d, ..]     (terms u + d >= n dropped)
//! sel[u][d]    = [start == d] * [d + u < stop]
//! ```
//!
//! applied to one axis at a time (rows, then columns, ...).
//!
//! **Preconditions.** Every axis has step 1 and `0 <= start <= n` (Python's
//! slice normalization established it before the placeholder was emitted);
//! `stop` is any integer. A start or stop encoding, when given, is exact
//! (`scan.rs`). `padding_zero` only when every source cell at a position
//! `>= stop` of the axis is 0 (`constructions::padding_is_zero`).
//!
//! **Postconditions.** `y` holds the window `x[start:stop]` at the top-left
//! of a buffer of `x`'s shape and 0 everywhere else: the cells the
//! element-wise lowering produced, so the recorded extents and the output
//! layout are unchanged.
//!
//! **Soundness.** No prover hint: every selector is a product of exact
//! indicator bits. Since `start` is in `[0, n]`, at most one `[start == d]`
//! is 1, so each sum is either `x[.., start + u, ..]` (when `start + u <
//! stop` and `start + u < n`) or 0. Without the mask (`padding_zero`), a
//! position past `stop` reads a source cell at a position `>= stop`, which is
//! 0 by the precondition; the shifts along the other axes only combine cells
//! of one position of this axis, so that holds for every intermediate array.
//!
//! **Cost** (`cost::window_shift`) for an `R x C` window: `R(R+1)/2 C +
//! C(C+1)/2 R` multiply-adds (3 halo2-base cells, 1 Noir gate each), plus per
//! axis the selectors: equality tests unless the start's one-hot is given,
//! and the mask (one product per selector, plus a comparison per position
//! without a thermometer of `stop`; nothing when the padding is 0).

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, row_major_strides};
use crate::types::{NumberType, ScalarValue, Value};

use super::encoding::{onehot_over, IndexEncoding};

/// One windowed axis: `start:stop` with step 1, `start` in `[0, n]`.
pub struct AxisWindow {
    pub axis: usize,
    pub start: Value,
    pub stop: Value,
    /// The encoding of `start`, when a scan produced it.
    pub start_encoding: Option<IndexEncoding>,
    /// The encoding of `stop`, when it has a thermometer.
    pub stop_encoding: Option<IndexEncoding>,
    /// Every source cell past `stop` on this axis is 0: no mask.
    pub padding_zero: bool,
}

/// The window of `cells` (row-major over `shape`, integer or float) given by
/// `windows`, at the top-left of a buffer of `shape` padded with 0.
pub fn window_shift(
    b: &mut IRBuilder,
    cells: Vec<Value>,
    shape: &[usize],
    windows: &[AxisWindow],
    dtype: NumberType,
) -> Vec<Value> {
    let strides = row_major_strides(shape);
    let mut cells = cells;
    for window in windows {
        let selectors = axis_selectors(b, shape[window.axis], window);
        cells = shift_axis(b, &cells, shape, &strides, window.axis, &selectors, dtype);
    }
    cells
}

/// `sel[u][d] = [start == d] * [d + u < stop]` for `u + d < n` (the second
/// factor dropped when it is always 1 on the selected terms or the padding
/// is 0 anyway).
fn axis_selectors(b: &mut IRBuilder, n: usize, window: &AxisWindow) -> Vec<Vec<Value>> {
    let at = onehot_over(b, &window.start, window.start_encoding.as_ref(), 0, n);
    let mask = StopMask::new(b, n, window);
    (0..n)
        .map(|u| (0..n - u).map(|d| mask.select(b, &at[d], d, u)).collect())
        .collect()
}

/// How a selector learns `[d + u < stop]` (for the `d` whose bit is set).
enum StopMask {
    /// `stop >= n` (every kept term is inside the window), or the cells past
    /// `stop` are 0.
    Unneeded,
    /// A thermometer of `stop`: the bit is free.
    Thermometer(IndexEncoding),
    /// One comparison per output position, `[start + u < stop]`.
    Compared(Vec<Value>),
}

impl StopMask {
    fn new(b: &mut IRBuilder, n: usize, window: &AxisWindow) -> StopMask {
        if window.padding_zero || window.stop.int_val().is_some_and(|s| s >= n as i64) {
            return StopMask::Unneeded;
        }
        if let Some(enc) = window.stop_encoding.clone().filter(|e| e.below.is_some()) {
            return StopMask::Thermometer(enc);
        }
        let masks = (0..n)
            .map(|u| {
                let u_v = b.ir_constant_int(u as i64);
                let position = b.ir_add_i(&window.start, &u_v);
                b.ir_less_than_i_bounded(&position, &window.stop)
            })
            .collect();
        StopMask::Compared(masks)
    }

    /// `at_d * [d + u < stop]`, given `at_d = [start == d]`.
    fn select(&self, b: &mut IRBuilder, at_d: &Value, d: usize, u: usize) -> Value {
        match self {
            StopMask::Unneeded => at_d.clone(),
            StopMask::Compared(masks) => b.ir_mul_i(at_d, &masks[u]),
            StopMask::Thermometer(enc) => {
                // [t < stop] = 1 - [stop < t + 1].
                let stop_le_t = enc.less_than(b, (d + u) as i64 + 1).expect("the encoding has a thermometer");
                stop_le_t.not().and(b, at_d)
            }
        }
    }
}

/// One shift along `axis`: `y[.., u, ..] = sum_d sel[u][d] * x[.., u + d, ..]`.
fn shift_axis(
    b: &mut IRBuilder,
    cells: &[Value],
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    selectors: &[Vec<Value>],
    dtype: NumberType,
) -> Vec<Value> {
    (0..cells.len())
        .map(|flat| {
            let u = decode_coords(flat, shape, strides)[axis];
            let terms: Vec<Value> = selectors[u]
                .iter()
                .enumerate()
                .map(|(d, sel)| scale_by_bit(b, sel, &cells[flat + d * strides[axis]], dtype))
                .collect();
            sum_typed(b, terms, dtype)
        })
        .collect()
}

/// `bit * x` for a 0/1 integer `bit`. For a float `x` this is the integer
/// product of the bit and `x`'s fixed-point code, which is the code of the
/// real product exactly (no rescale), so it is emitted as `MulI` and read
/// back as a float.
pub(crate) fn scale_by_bit(b: &mut IRBuilder, bit: &Value, x: &Value, dtype: NumberType) -> Value {
    let product = b.ir_mul_i(bit, x);
    match (dtype, product) {
        (NumberType::Float, Value::Integer(p)) => Value::Float(ScalarValue {
            static_val: None,
            stmt_id: p.stmt_id,
            value_id: p.value_id,
        }),
        (_, product) => product,
    }
}

/// The sum of `terms` with the addition of `dtype` (`AddF` is the field
/// addition `AddI` on every backend).
pub(crate) fn sum_typed(b: &mut IRBuilder, terms: Vec<Value>, dtype: NumberType) -> Value {
    let mut terms = terms.into_iter();
    let first = terms.next().expect("at least one term (u < n)");
    terms.fold(first, |acc, t| match dtype {
        NumberType::Float => b.ir_add_f(&acc, &t),
        _ => b.ir_add_i(&acc, &t),
    })
}
