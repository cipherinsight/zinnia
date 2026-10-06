//! Window gather: `x[s0:e0, s1:e1, ...]` with runtime bounds as a shift of
//! the whole array by a shared one-hot per axis.
//!
//! The element-wise lowering reads every output cell on its own: per output
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
//! **Preconditions.** Every windowed axis has step 1 and `0 <= start <= n`
//! (Python's slice normalization established it, `fast` in the caller);
//! `stop` is any integer (normalized into `[0, n]`).
//!
//! **Postconditions.** `y` holds the window `x[start:stop]` at the top-left
//! of a buffer of `x`'s shape and the default 0 everywhere else: the same
//! cells the element-wise lowering produces, so the recorded extents and
//! the output layout are unchanged.
//!
//! **Soundness.** No prover hint: every selector is a product of indicator
//! bits, which are exact (equality tests, or a recorded `IndexEncoding`
//! whose bits are exact under its producer's precondition). Since `start`
//! is in `[0, n]`, at most one `[start == d]` is 1, so each sum is either
//! `x[.., start + u, ..]` (when `start + u < stop` and `start + u < n`) or 0,
//! which is the slice's value or its padding.
//!
//! **Cost** for an `R x C` window (no encoding available): halo2-base
//! `R + C` equality tests and comparisons for the selectors, `R(R+1)/2 +
//! C(C+1)/2` selector products, and `(R(R+1)/2) C + (C(C+1)/2) R`
//! multiply-adds (3 cells each); with a recorded encoding the equality
//! tests and comparisons are free. Noir: one gate per product.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, row_major_strides};
use crate::types::{NumberType, ScalarValue, Value};

use super::encoding::{encoding_of, onehot_over, Bit, IndexEncoding};
use super::vanishing::forced_zero_by_unset;

/// One windowed axis: `start:stop` with step 1, `start` in `[0, n]`.
pub struct AxisWindow {
    pub axis: usize,
    pub start: Value,
    pub stop: Value,
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
    let source = cells;
    let mut cells = source.clone();
    for window in windows {
        let padding_zero = padding_is_zero(b, &source, shape, &strides, window);
        let selectors = axis_selectors(b, shape[window.axis], window, padding_zero);
        cells = shift_axis(b, &cells, shape, &strides, window.axis, &selectors, dtype);
    }
    cells
}

/// Whether every source cell at an axis position past `stop` is 0, so the
/// window's padding needs no mask: `stop` is the last set position of flags
/// plus at least 1 (a recorded encoding), and each flag being 0 forces the
/// source cells at its position to 0 (`vanishing`). The shift along other
/// axes only combines cells of the same position on this one, so the
/// property holds for the partially shifted array too.
fn padding_is_zero(
    b: &IRBuilder,
    source: &[Value],
    shape: &[usize],
    strides: &[usize],
    window: &AxisWindow,
) -> bool {
    let Some(enc) = encoding_of(b, &window.stop) else { return false };
    let Some(flags) = enc.last_of.as_ref() else { return false };
    if enc.lo < 1 || flags.len() != shape[window.axis] {
        return false;
    }
    flags.iter().enumerate().all(|(k, flag)| {
        let Some(flag) = flag.stmt_id() else { return false };
        let zeros = forced_zero_by_unset(&b.stmts, flag);
        source.iter().enumerate().all(|(flat, cell)| {
            decode_coords(flat, shape, strides)[window.axis] != k
                || cell.stmt_id().is_some_and(|c| zeros.contains(&c))
        })
    })
}

/// `sel[u][d] = [start == d] * [d + u < stop]` for `u + d < n` (the second
/// factor dropped when the padding is 0 anyway).
fn axis_selectors(b: &mut IRBuilder, n: usize, window: &AxisWindow, padding_zero: bool) -> Vec<Vec<Value>> {
    let at = onehot_over(b, &window.start, 0, n);
    let mask = if padding_zero {
        StopMask::Unneeded
    } else {
        StopMask::new(b, n, &window.start, &window.stop)
    };
    (0..n)
        .map(|u| (0..n - u).map(|d| mask.select(b, &at[d], d, u)).collect())
        .collect()
}

/// How a selector learns `[d + u < stop]` (for the `d` whose bit is set).
enum StopMask {
    /// `stop >= n` (every kept term is inside the window), or the cells
    /// past `stop` are 0 (`padding_is_zero`).
    Unneeded,
    /// A thermometer of `stop` (a recorded encoding): the bit is free.
    Thermometer(IndexEncoding),
    /// One comparison per output position, `[start + u < stop]`.
    Compared(Vec<Value>),
}

impl StopMask {
    fn new(b: &mut IRBuilder, n: usize, start: &Value, stop: &Value) -> StopMask {
        if stop.int_val().is_some_and(|s| s >= n as i64) {
            return StopMask::Unneeded;
        }
        if let Some(enc) = encoding_of(b, stop).filter(|e| e.below.is_some()) {
            return StopMask::Thermometer(enc);
        }
        let masks = (0..n)
            .map(|u| {
                let u_v = b.ir_constant_int(u as i64);
                let position = b.ir_add_i(start, &u_v);
                b.ir_less_than_i_bounded(&position, stop)
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
                let stop_le_t = enc.less_than(b, (d + u) as i64 + 1).expect("encoding has a thermometer");
                Bit { value: stop_le_t.value, negated: !stop_le_t.negated }.and(b, at_d)
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
