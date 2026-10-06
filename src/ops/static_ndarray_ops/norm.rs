//! `np.linalg.norm(x, ord=None, axis=None, keepdims=False)` on static arrays.
//!
//! Follows NumPy's definition (`numpy/linalg/_linalg.py`) for the orders
//! that need no singular values:
//!
//! * vector norms (one reduced axis): `None`/`2` = `sqrt(sum |x|^2)`,
//!   `1` = `sum |x|`, `inf` = `max |x|`, `-inf` = `min |x|`,
//!   `0` = the number of non-zero elements;
//! * matrix norms (two reduced axes): `None`/`'fro'` = `sqrt(sum |x|^2)`,
//!   `1` / `-1` = max / min over columns of the column sums of `|x|`,
//!   `inf` / `-inf` = max / min over rows of the row sums of `|x|`;
//! * `axis=None` with `ord=None` is the 2-norm of the flattened array, of
//!   any rank; with another `ord` the array must be 1-D or 2-D.
//!
//! The result is always a float (NumPy converts integer input to float).
//! Every other order (`'nuc'`, matrix `2`/`-2`, which need singular values,
//! and general vector `p`-norms, which need a fractional power) is
//! rejected with an error that names it.
//!
//! Sums of squares are one fused `ProductSumF` (one rescale, as `np.dot`),
//! so `sqrt` sees the real sum rounded once.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

/// The `ord` argument of `np.linalg.norm`, as a compile-time constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormOrder {
    /// `ord=None`.
    Default,
    /// `ord='fro'`.
    Frobenius,
    /// `ord='nuc'`.
    Nuclear,
    /// `ord=inf`.
    Inf,
    /// `ord=-inf`.
    NegInf,
    /// A finite number.
    Number(f64),
}

impl std::fmt::Display for NormOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormOrder::Default => write!(f, "None"),
            NormOrder::Frobenius => write!(f, "'fro'"),
            NormOrder::Nuclear => write!(f, "'nuc'"),
            NormOrder::Inf => write!(f, "inf"),
            NormOrder::NegInf => write!(f, "-inf"),
            NormOrder::Number(p) => write!(f, "{p}"),
        }
    }
}

/// The bound arguments of one `np.linalg.norm` call.
pub struct NormCall {
    pub x: Value,
    pub ord: NormOrder,
    /// `None` for `axis=None`; otherwise the listed axes (possibly negative).
    pub axis: Option<Vec<i64>>,
    pub keepdims: bool,
}

/// What one output element reduces.
enum Reduction {
    /// A vector norm along one axis.
    Vector { axis: usize },
    /// A matrix norm over `(row_axis, col_axis)`.
    Matrix { row_axis: usize, col_axis: usize },
    /// The 2-norm of every element (`axis=None`, `ord=None`).
    Flat,
}

pub fn np_linalg_norm(b: &mut IRBuilder, call: NormCall) -> Value {
    let x = match &call.x {
        Value::StaticArray { .. } => call.x.clone(),
        Value::List(_) | Value::Tuple(_) => crate::helpers::static_array::base::to_static_array(b, &call.x)
            .unwrap_or_else(|| panic!("np.linalg.norm: x must be a rectangular array of numbers")),
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => {
            return scalar_norm(b, &call);
        }
        Value::DynamicNDArray(_) => panic!("np.linalg.norm is not yet supported for bounded-dynamic arrays"),
        other => panic!("np.linalg.norm: unsupported argument type {}", other.zinnia_type()),
    };
    let Value::StaticArray { shape, dtype, .. } = &x else { unreachable!() };
    let (shape, dtype) = (shape.clone(), *dtype);
    let reduction = resolve_reduction(&call, shape.len());
    let cells = crate::helpers::static_array::elementwise::payload_cells(b, &x);
    let reduced_axes: Vec<usize> = match reduction {
        Reduction::Vector { axis } => vec![axis],
        Reduction::Matrix { row_axis, col_axis } => vec![row_axis, col_axis],
        Reduction::Flat => (0..shape.len()).collect(),
    };
    let groups = reduction_groups(&shape, &reduced_axes);
    let mut out = Vec::with_capacity(groups.len());
    for group in &groups {
        let elements: Vec<Value> = group.iter().map(|&i| cells[i].clone()).collect();
        out.push(match reduction {
            Reduction::Vector { .. } => vector_norm(b, call.ord, &elements, dtype),
            Reduction::Flat => two_norm(b, &elements),
            Reduction::Matrix { row_axis, col_axis } => {
                // `elements` is row-major over (row_axis, col_axis).
                let (rows, cols) = (shape[row_axis], shape[col_axis]);
                matrix_norm(b, call.ord, &elements, rows, cols)
            }
        });
    }
    let out_shape: Vec<usize> = (0..shape.len())
        .filter_map(|axis| {
            if !reduced_axes.contains(&axis) {
                Some(shape[axis])
            } else if call.keepdims {
                Some(1)
            } else {
                None
            }
        })
        .collect();
    if out_shape.is_empty() {
        return out.pop().expect("one output element");
    }
    crate::helpers::static_array::base::build_static_array_from_flat(b, out, out_shape, NumberType::Float)
}

/// The norm of a 0-d `x`: NumPy allows only `ord=None, axis=None`, which is `|x|`.
fn scalar_norm(b: &mut IRBuilder, call: &NormCall) -> Value {
    if call.ord != NormOrder::Default || call.axis.is_some() {
        panic!("np.linalg.norm: Improper number of dimensions to norm (a scalar takes neither ord nor axis)");
    }
    if call.keepdims {
        panic!("np.linalg.norm: keepdims=True on a scalar is not supported");
    }
    abs_float(b, &call.x)
}

/// Decide which axes one output element reduces, with NumPy's checks.
fn resolve_reduction(call: &NormCall, ndim: usize) -> Reduction {
    let normalize = |axis: i64| -> usize {
        let resolved = if axis < 0 { axis + ndim as i64 } else { axis };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!("np.linalg.norm: axis {axis} is out of bounds for array of dimension {ndim}");
        }
        resolved as usize
    };
    match &call.axis {
        None if call.ord == NormOrder::Default => Reduction::Flat,
        None => match ndim {
            1 => Reduction::Vector { axis: 0 },
            2 => Reduction::Matrix { row_axis: 0, col_axis: 1 },
            _ => panic!("np.linalg.norm: Improper number of dimensions to norm (ord={} needs a 1-D or 2-D array or an axis)", call.ord),
        },
        Some(axes) => match axes.as_slice() {
            [axis] => Reduction::Vector { axis: normalize(*axis) },
            [row, col] => {
                let (row_axis, col_axis) = (normalize(*row), normalize(*col));
                if row_axis == col_axis {
                    panic!("np.linalg.norm: Duplicate axes given");
                }
                Reduction::Matrix { row_axis, col_axis }
            }
            _ => panic!("np.linalg.norm: Improper number of dimensions to norm (axis must name 1 or 2 axes)"),
        },
    }
}

/// For every output position (row-major over the axes not in `axes`), the
/// flat indices of the elements it reduces, row-major over `axes` in the
/// order given.
fn reduction_groups(shape: &[usize], axes: &[usize]) -> Vec<Vec<usize>> {
    let strides = crate::helpers::shape_arith::row_major_strides(shape);
    let kept: Vec<usize> = (0..shape.len()).filter(|a| !axes.contains(a)).collect();
    let inner = flat_offsets(shape, &strides, axes);
    flat_offsets(shape, &strides, &kept)
        .into_iter()
        .map(|base| inner.iter().map(|&o| base + o).collect())
        .collect()
}

/// The flat offsets of every index combination of `dims`, row-major in
/// the order of `dims`.
fn flat_offsets(shape: &[usize], strides: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut acc = vec![0usize];
    for &axis in dims {
        acc = acc
            .iter()
            .flat_map(|&base| (0..shape[axis]).map(move |i| base + i * strides[axis]))
            .collect();
    }
    acc
}

fn vector_norm(b: &mut IRBuilder, ord: NormOrder, elements: &[Value], dtype: NumberType) -> Value {
    match ord {
        NormOrder::Default => two_norm(b, elements),
        NormOrder::Number(p) if p == 2.0 => two_norm(b, elements),
        NormOrder::Number(p) if p == 1.0 => {
            let magnitudes = abs_floats(b, elements);
            sum_floats(b, &magnitudes)
        }
        NormOrder::Number(p) if p == 0.0 => {
            if dtype == NumberType::Complex {
                panic!("np.linalg.norm: ord=0 of a complex array is not supported");
            }
            let indicators: Vec<Value> = elements
                .iter()
                .map(|x| {
                    let zero = b.ir_constant_int(0);
                    let nonzero = crate::helpers::value_ops::apply_binary_op(b, "ne", x, &zero);
                    b.ir_float_cast(&nonzero)
                })
                .collect();
            sum_floats(b, &indicators)
        }
        NormOrder::Inf | NormOrder::NegInf => {
            let magnitudes = abs_floats(b, elements);
            extreme(b, if ord == NormOrder::Inf { "max" } else { "min" }, magnitudes)
        }
        NormOrder::Frobenius | NormOrder::Nuclear => {
            panic!("np.linalg.norm: Invalid norm order {ord} for vectors")
        }
        NormOrder::Number(_) => panic!(
            "np.linalg.norm: vector ord={ord} is not supported (it needs a fractional power); \
             supported vector orders are None, 2, 1, 0, inf and -inf"
        ),
    }
}

/// `elements` row-major over a `rows x cols` matrix.
fn matrix_norm(b: &mut IRBuilder, ord: NormOrder, elements: &[Value], rows: usize, cols: usize) -> Value {
    match ord {
        NormOrder::Default | NormOrder::Frobenius => two_norm(b, elements),
        NormOrder::Number(p) if p == 1.0 || p == -1.0 => {
            let magnitudes = abs_floats(b, elements);
            let column_sums: Vec<Value> = (0..cols)
                .map(|c| {
                    let column: Vec<Value> = (0..rows).map(|r| magnitudes[r * cols + c].clone()).collect();
                    sum_floats(b, &column)
                })
                .collect();
            extreme(b, if p > 0.0 { "max" } else { "min" }, column_sums)
        }
        NormOrder::Inf | NormOrder::NegInf => {
            let magnitudes = abs_floats(b, elements);
            let row_sums: Vec<Value> = magnitudes.chunks(cols.max(1)).take(rows).map(|row| sum_floats(b, row)).collect();
            extreme(b, if ord == NormOrder::Inf { "max" } else { "min" }, row_sums)
        }
        NormOrder::Nuclear => panic!("np.linalg.norm: ord='nuc' is not supported (it needs singular values)"),
        NormOrder::Number(p) if p == 2.0 || p == -2.0 => {
            panic!("np.linalg.norm: matrix ord={ord} is not supported (it needs singular values)")
        }
        NormOrder::Number(_) => panic!("np.linalg.norm: Invalid norm order {ord} for matrices"),
    }
}

/// `sqrt(sum |x|^2)` with the squares summed exactly and rescaled once.
fn two_norm(b: &mut IRBuilder, elements: &[Value]) -> Value {
    let mut pairs = Vec::with_capacity(elements.len());
    for x in elements {
        match x {
            Value::Complex { real, imag } => {
                let (re, im) = (Value::Float(real.clone()), Value::Float(imag.clone()));
                pairs.push((re.clone(), re));
                pairs.push((im.clone(), im));
            }
            _ => {
                let xf = as_float(b, x);
                pairs.push((xf.clone(), xf));
            }
        }
    }
    if pairs.is_empty() {
        return b.ir_constant_float(0.0);
    }
    let sum_of_squares = b.ir_product_sum_f(&pairs);
    b.ir_sqrt_f(&sum_of_squares)
}

fn abs_floats(b: &mut IRBuilder, elements: &[Value]) -> Vec<Value> {
    elements.iter().map(|x| abs_float(b, x)).collect()
}

/// `|x|` as a float (the modulus of a complex number).
fn abs_float(b: &mut IRBuilder, x: &Value) -> Value {
    match x {
        Value::Complex { real, imag } => {
            let (re, im) = (Value::Float(real.clone()), Value::Float(imag.clone()));
            let sum_of_squares = b.ir_product_sum_f(&[(re.clone(), re), (im.clone(), im)]);
            b.ir_sqrt_f(&sum_of_squares)
        }
        _ => {
            let xf = as_float(b, x);
            b.ir_abs_f(&xf)
        }
    }
}

fn as_float(b: &mut IRBuilder, x: &Value) -> Value {
    match x {
        Value::Float(_) => x.clone(),
        Value::Integer(_) | Value::Boolean(_) => b.ir_float_cast(x),
        other => panic!("np.linalg.norm: unsupported element type {}", other.zinnia_type()),
    }
}

fn sum_floats(b: &mut IRBuilder, values: &[Value]) -> Value {
    if values.is_empty() {
        return b.ir_constant_float(0.0);
    }
    b.ir_sum_f(values)
}

/// The maximum or minimum of `values` (NumPy raises on an empty reduction).
fn extreme(b: &mut IRBuilder, op: &str, values: Vec<Value>) -> Value {
    if values.is_empty() {
        panic!("np.linalg.norm: zero-size array to reduction operation {op}imum which has no identity");
    }
    let list = Value::List(crate::types::CompositeData {
        elements_type: values.iter().map(|v| v.zinnia_type()).collect(),
        values,
        value_id: crate::types::ValueId::next(),
    });
    crate::helpers::ndarray::builtin_reduce(b, op, &list)
}

#[cfg(test)]
mod tests {
    use super::reduction_groups;

    #[test]
    fn groups_follow_numpy_axis_order() {
        // shape (2, 3): reducing axis 1 gives one group per row.
        assert_eq!(reduction_groups(&[2, 3], &[1]), vec![vec![0, 1, 2], vec![3, 4, 5]]);
        // reducing axis 0 gives one group per column.
        assert_eq!(reduction_groups(&[2, 3], &[0]), vec![vec![0, 3], vec![1, 4], vec![2, 5]]);
        // (row, col) = (1, 0): row-major over axis 1 then axis 0.
        assert_eq!(reduction_groups(&[2, 3], &[1, 0]), vec![vec![0, 3, 1, 4, 2, 5]]);
        // shape (2, 2, 2), reducing the middle axis.
        assert_eq!(reduction_groups(&[2, 2, 2], &[1]), vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]);
    }
}
