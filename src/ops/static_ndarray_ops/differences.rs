//! Differences of consecutive elements: `np.ediff1d` and the trapezoidal
//! rule `np.trapezoid`.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{resolve_axis, row_major_strides};
use crate::types::{NumberType, Value};

use super::operand::{build_array, cell_op, StaticOperand};

/// `np.ediff1d(ary, to_end=None, to_begin=None)`: the differences
/// `ary.flat[i + 1] - ary.flat[i]`, with the values of `to_begin` before
/// them and those of `to_end` after (both flattened). The result has
/// `ary`'s dtype; like NumPy, a float `to_begin` / `to_end` for an integer
/// array is rejected (it cannot be cast with `same_kind`).
pub fn np_ediff1d(b: &mut IRBuilder, ary: &Value, to_end: Option<&Value>, to_begin: Option<&Value>) -> Value {
    let ary = StaticOperand::read_numeric(b, ary, "np.ediff1d");
    let mut extra = |value: Option<&Value>, name: &str| -> Vec<Value> {
        let Some(value) = value.filter(|v| !matches!(v, Value::None)) else {
            return Vec::new();
        };
        let operand = StaticOperand::read(b, value, "np.ediff1d");
        if operand.dtype == NumberType::Float && ary.dtype != NumberType::Float {
            panic!("np.ediff1d: dtype of `{name}` must be compatible with input `ary` under the `same_kind` rule");
        }
        operand.cells
    };
    let before = extra(to_begin, "to_begin");
    let after = extra(to_end, "to_end");
    let mut cells = before;
    for pair in ary.cells.windows(2) {
        cells.push(cell_op(b, "sub", &pair[1], &pair[0]));
    }
    cells.extend(after);
    let length = cells.len();
    build_array(b, cells, vec![length], ary.dtype)
}

/// The spacing of the samples of `np.trapezoid`.
pub enum TrapezoidSpacing<'a> {
    /// The sample points `x`: a 1-D array along the integration axis, or an
    /// array of `y`'s shape.
    Points(&'a Value),
    /// A constant spacing `dx` (NumPy's default is 1.0).
    Constant(&'a Value),
}

/// `np.trapezoid(y, x=None, dx=1.0, axis=-1)`: the integral of `y` along
/// `axis` by the trapezoidal rule, as NumPy computes it: the sum over the
/// intervals of `d * (y[i + 1] + y[i]) / 2.0`, where `d` is `x[i + 1] -
/// x[i]` or `dx`. The result is a float with `y`'s other axes.
pub fn np_trapezoid(b: &mut IRBuilder, y: &Value, spacing: TrapezoidSpacing, axis: i64) -> Value {
    let y = StaticOperand::read_numeric(b, y, "np.trapezoid");
    if y.shape.is_empty() {
        panic!("np.trapezoid: y must have at least one dimension");
    }
    let axis = resolve_axis(axis, y.shape.len(), "np.trapezoid");
    let length = y.shape[axis];
    let strides = row_major_strides(&y.shape);
    let step = strides[axis];

    let x = match &spacing {
        TrapezoidSpacing::Points(x) => {
            let x = StaticOperand::read_numeric(b, x, "np.trapezoid");
            if x.shape != [length] && x.shape != y.shape {
                panic!(
                    "np.trapezoid: x must be 1-D of length {length} or have y's shape {:?}, got shape {:?}",
                    y.shape, x.shape
                );
            }
            Some(x)
        }
        TrapezoidSpacing::Constant(_) => None,
    };
    // The spacing of interval `i` of the line starting at `base`.
    let interval = |b: &mut IRBuilder, base: usize, i: usize| -> Value {
        match (&x, &spacing) {
            (Some(x), _) if x.shape.len() == 1 => cell_op(b, "sub", &x.cells[i + 1], &x.cells[i]),
            (Some(x), _) => cell_op(b, "sub", &x.cells[base + (i + 1) * step], &x.cells[base + i * step]),
            (None, TrapezoidSpacing::Constant(dx)) => (*dx).clone(),
            (None, TrapezoidSpacing::Points(_)) => unreachable!("sample points are read above"),
        }
    };

    let mut out_shape = y.shape.clone();
    out_shape.remove(axis);
    let lines: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);
    let two = b.ir_constant_float(2.0);
    let mut cells = Vec::with_capacity(lines);
    for line in 0..lines {
        // The position in `y` of the line's first sample.
        let base: usize = (0..y.shape.len())
            .filter(|&d| d != axis)
            .zip(0..)
            .map(|(d, k)| line / out_strides[k] % out_shape[k] * strides[d])
            .sum();
        let mut total = b.ir_constant_float(0.0);
        for i in 0..length.saturating_sub(1) {
            let pair = cell_op(b, "add", &y.cells[base + (i + 1) * step], &y.cells[base + i * step]);
            let d = interval(b, base, i);
            let area = cell_op(b, "mul", &d, &pair);
            let area = cell_op(b, "div", &area, &two);
            total = cell_op(b, "add", &total, &area);
        }
        cells.push(total);
    }
    if out_shape.is_empty() {
        return cells.pop().expect("a 1-D integral has one value");
    }
    build_array(b, cells, out_shape, NumberType::Float)
}
