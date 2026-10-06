//! Numpy / math / zinnia / DynamicNDArray namespaced function dispatch.
//!
//! Split into two halves:
//!   - `try_numpy_func`: explicit named-arm matches for `np.zeros`,
//!     `np.concatenate`, `np.split`, etc. plus `DynamicNDArray.zeros` /
//!     `zinnia.zeros` style class-method calls.
//!   - `try_np_fallback`: the catch-all `(Some("np"|"zinnia"|"math"), name)`
//!     arm that delegates to the `ops::registry` and the centralised
//!     unary/binary vectorisation paths.

use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, NumberType, ScalarValue, Value, ValueId, ZinniaType};

use super::super::IRGenerator;
use super::DispatchCtx;

fn ensure_no_kwargs(op: &str, kwargs: &HashMap<String, Value>, supported: &[&str]) {
    for key in kwargs.keys() {
        if !supported.iter().any(|allowed| allowed == key) {
            panic!("{}: keyword argument `{}` is not supported", op, key);
        }
    }
}

fn numeric_leaf_count(v: &Value) -> Option<usize> {
    match v {
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => Some(1),
        Value::List(d) | Value::Tuple(d) => d
            .values
            .iter()
            .try_fold(0usize, |acc, elem| Some(acc + numeric_leaf_count(elem)?)),
        Value::StaticArray { shape, .. } => Some(shape.iter().product()),
        _ => None,
    }
}

fn np_array_equal(b: &mut crate::builder::IRBuilder, args: &[Value]) -> Value {
    assert!(args.len() >= 2, "np.array_equal: requires two input arrays");
    if matches!(args[0], Value::DynamicNDArray(_)) || matches!(args[1], Value::DynamicNDArray(_)) {
        panic!(
            "np.array_equal is not yet supported for bounded-dynamic arrays; active-lane equality needs runtime-shape-aware semantics"
        );
    }
    let lhs_count = numeric_leaf_count(&args[0]);
    let rhs_count = numeric_leaf_count(&args[1]);
    if lhs_count != rhs_count {
        return b.ir_constant_bool(false);
    }
    let cmp = crate::helpers::value_ops::apply_binary_op(b, "eq", &args[0], &args[1]);
    crate::helpers::value_ops::to_scalar_bool(b, &cmp)
}

/// `np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`:
/// `|a - b| <= atol + rtol * |b|` element-wise, computed in Q32 fixed point
/// (integers are converted to floats, as NumPy does, so the tolerance term
/// is rounded to Q32 like any float constant).
///
/// `equal_nan` must be a compile-time boolean. Zinnia's numbers have no NaN
/// (inputs are checked to be finite Q32 values and every operation yields a
/// number), so over the values Zinnia computes, `equal_nan=True` and `False`
/// give the same result.
fn np_isclose(
    b: &mut crate::builder::IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    ensure_no_kwargs("np.isclose", kwargs, &["rtol", "atol", "equal_nan"]);
    if args.len() < 2 || args.len() > 5 {
        panic!("np.isclose takes 2 to 5 positional arguments (a, b, rtol, atol, equal_nan)");
    }
    if matches!(args[0], Value::DynamicNDArray(_)) || matches!(args[1], Value::DynamicNDArray(_)) {
        panic!(
            "np.isclose is not yet supported for bounded-dynamic arrays; active-lane tolerance comparisons need runtime-shape-aware semantics"
        );
    }
    let mut argument = |position: usize, name: &str| -> Option<Value> {
        match (args.get(position), kwargs.get(name)) {
            (Some(_), Some(_)) => {
                panic!("np.isclose got multiple values for argument '{}'", name)
            }
            (Some(v), None) | (None, Some(v)) => Some(v.clone()),
            (None, None) => None,
        }
    };
    let rtol = argument(2, "rtol").unwrap_or_else(|| b.ir_constant_float(1e-05));
    let atol = argument(3, "atol").unwrap_or_else(|| b.ir_constant_float(1e-08));
    if let Some(equal_nan) = argument(4, "equal_nan") {
        if equal_nan.bool_val().or_else(|| equal_nan.int_val().map(|v| v != 0)).is_none() {
            panic!("np.isclose: equal_nan must be a compile-time True or False");
        }
    }
    let lhs = &args[0];
    let rhs = &args[1];
    let diff = crate::helpers::value_ops::apply_binary_op(b, "sub", lhs, rhs);
    let abs_diff = np_abs_value(b, &diff);
    let abs_rhs = np_abs_value(b, rhs);
    let scaled = crate::helpers::value_ops::apply_binary_op(b, "mul", &rtol, &abs_rhs);
    let tol = crate::helpers::value_ops::apply_binary_op(b, "add", &atol, &scaled);
    crate::helpers::value_ops::apply_binary_op(b, "lte", &abs_diff, &tol)
}

fn np_abs_value(b: &mut crate::builder::IRBuilder, v: &Value) -> Value {
    match v {
        Value::StaticArray { .. } => {
            if matches!(
                v,
                Value::StaticArray {
                    dtype: NumberType::Complex,
                    ..
                }
            ) {
                crate::helpers::static_array::complex::np_abs_complex_static_array(b, v)
            } else {
                let (shape, dtype) = match v {
                    Value::StaticArray { shape, dtype, .. } => (shape.clone(), *dtype),
                    _ => unreachable!(),
                };
                let cells = crate::helpers::static_array::elementwise::payload_cells(b, v);
                let out: Vec<Value> = cells
                    .iter()
                    .map(|cell| match cell {
                        Value::Float(_) => b.ir_abs_f(cell),
                        Value::Integer(_) | Value::Boolean(_) => b.ir_abs_i(cell),
                        _ => panic!("np.isclose: unsupported StaticArray element type"),
                    })
                    .collect();
                crate::helpers::static_array::base::build_static_array_from_flat(
                    b, out, shape, dtype,
                )
            }
        }
        Value::List(d) | Value::Tuple(d) => {
            let vals: Vec<Value> = d.values.iter().map(|elem| np_abs_value(b, elem)).collect();
            let types = vals.iter().map(|elem| elem.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: vals,
                value_id: ValueId::next(),
            })
        }
        Value::Float(_) => b.ir_abs_f(v),
        Value::Integer(_) | Value::Boolean(_) => b.ir_abs_i(v),
        Value::Complex { real, imag } => {
            let r = Value::Float(real.clone());
            let i = Value::Float(imag.clone());
            let rr = b.ir_mul_f(&r, &r);
            let ii = b.ir_mul_f(&i, &i);
            let sum = b.ir_add_f(&rr, &ii);
            b.ir_sqrt_f(&sum)
        }
        _ => panic!("np.isclose: unsupported operand type {}", v.zinnia_type()),
    }
}

fn static_array_meta(val: &Value, op: &str) -> (NumberType, Vec<usize>) {
    match val {
        Value::StaticArray { dtype, shape, .. } => (*dtype, shape.clone()),
        _ => panic!("{op}: this DS1000 slice supports StaticArray inputs only"),
    }
}

fn static_cells(b: &mut crate::builder::IRBuilder, val: &Value, op: &str) -> Vec<Value> {
    match val {
        Value::StaticArray { .. } => {
            crate::helpers::static_array::elementwise::payload_cells(b, val)
        }
        Value::List(_) | Value::Tuple(_) => {
            crate::helpers::composite::flatten_composite_with_builder(b, val)
        }
        _ => panic!("{op}: expected a static array/list"),
    }
}

fn as_float_value(b: &mut crate::builder::IRBuilder, val: &Value) -> Value {
    match val {
        Value::Float(_) => val.clone(),
        _ => b.ir_float_cast(val),
    }
}

fn product_sum(b: &mut crate::builder::IRBuilder, lhs: &[Value], rhs: &[Value]) -> Value {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "product_sum: operand lengths must match"
    );
    let pairs: Vec<(Value, Value)> = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(l, r)| (as_float_value(b, l), as_float_value(b, r)))
        .collect();
    b.ir_product_sum_f(&pairs)
}

fn np_percentile_ds1000(
    b: &mut crate::builder::IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    crate::ops::static_ndarray_ops::np_percentile(b, args, kwargs)
}

fn np_correlate_ds1000(
    b: &mut crate::builder::IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    ensure_no_kwargs("np.correlate", kwargs, &["mode"]);
    assert!(args.len() >= 2, "np.correlate: requires a and v");
    let mode = kwargs
        .get("mode")
        .or_else(|| args.get(2))
        .and_then(|v| v.string_val())
        .unwrap_or("valid");
    if mode != "valid" {
        panic!("np.correlate: DS1000 slice currently supports mode='valid' only");
    }
    let (lhs_dtype, lhs_shape) = static_array_meta(&args[0], "np.correlate");
    let (rhs_dtype, rhs_shape) = static_array_meta(&args[1], "np.correlate");
    if lhs_dtype == NumberType::Complex || rhs_dtype == NumberType::Complex {
        panic!("np.correlate: Complex inputs are not supported in this slice");
    }
    if lhs_shape.len() != 1 || rhs_shape.len() != 1 {
        panic!("np.correlate: DS1000 slice supports 1-D StaticArray inputs only");
    }
    let a = static_cells(b, &args[0], "np.correlate");
    let v = static_cells(b, &args[1], "np.correlate");
    // NumPy: `c[k] = sum_n a[n + k] * v[n]` when `len(a) >= len(v)`; when
    // `v` is longer it computes `correlate(v, a)` and returns it reversed.
    let swapped = v.len() > a.len();
    let (long, short) = if swapped { (&v, &a) } else { (&a, &v) };
    let out_len = long.len() - short.len() + 1;
    let integer_inputs = lhs_dtype == NumberType::Integer && rhs_dtype == NumberType::Integer;
    let mut out = Vec::with_capacity(out_len);
    for start in 0..out_len {
        let window = &long[start..start + short.len()];
        out.push(if integer_inputs {
            integer_product_sum(b, window, short)
        } else {
            product_sum(b, window, short)
        });
    }
    if swapped {
        out.reverse();
    }
    let out_dtype = if integer_inputs {
        NumberType::Integer
    } else {
        NumberType::Float
    };
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        out,
        vec![out_len],
        out_dtype,
    )
}

/// Exact integer dot product `sum_i lhs[i] * rhs[i]`.
fn integer_product_sum(b: &mut crate::builder::IRBuilder, lhs: &[Value], rhs: &[Value]) -> Value {
    let mut acc = b.ir_constant_int(0);
    for (l, r) in lhs.iter().zip(rhs) {
        let term = b.ir_mul_i(l, r);
        acc = b.ir_add_i(&acc, &term);
    }
    acc
}

fn np_corrcoef_ds1000(b: &mut crate::builder::IRBuilder, args: &[Value]) -> Value {
    crate::ops::static_ndarray_ops::np_corrcoef(b, args)
}

fn np_tensordot_ds1000(
    b: &mut crate::builder::IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    assert!(args.len() >= 2, "np.tensordot: requires a and b");
    let (a_dtype, a_shape) = static_array_meta(&args[0], "np.tensordot");
    let (b_dtype, b_shape) = static_array_meta(&args[1], "np.tensordot");
    if a_dtype == NumberType::Complex || b_dtype == NumberType::Complex {
        panic!("np.tensordot: Complex inputs are not supported in this slice");
    }
    let axes_value = kwargs
        .get("axes")
        .or_else(|| args.get(2))
        .unwrap_or_else(|| panic!("np.tensordot: requires axes"));
    let axes = match axes_value {
        Value::Tuple(d) | Value::List(d) if d.values.len() == 2 => {
            let lhs = match &d.values[0] {
                Value::Integer(_) => vec![d.values[0].int_val().unwrap()],
                Value::Tuple(td) | Value::List(td) => td
                    .values
                    .iter()
                    .map(|v| {
                        v.int_val()
                            .expect("np.tensordot: axes must be static integers")
                    })
                    .collect(),
                _ => panic!("np.tensordot: axes must be static integers"),
            };
            let rhs = match &d.values[1] {
                Value::Integer(_) => vec![d.values[1].int_val().unwrap()],
                Value::Tuple(td) | Value::List(td) => td
                    .values
                    .iter()
                    .map(|v| {
                        v.int_val()
                            .expect("np.tensordot: axes must be static integers")
                    })
                    .collect(),
                _ => panic!("np.tensordot: axes must be static integers"),
            };
            (lhs, rhs)
        }
        Value::Integer(_) => {
            let n = axes_value.int_val().unwrap();
            if n < 0 {
                panic!("np.tensordot: axes integer must be nonnegative");
            }
            let n = n as usize;
            let lhs: Vec<i64> = ((a_shape.len() - n)..a_shape.len())
                .map(|v| v as i64)
                .collect();
            let rhs: Vec<i64> = (0..n).map(|v| v as i64).collect();
            (lhs, rhs)
        }
        _ => panic!("np.tensordot: axes must be static"),
    };
    let norm_axis = |axis: i64, rank: usize, op: &str| -> usize {
        crate::helpers::shape_arith::resolve_axis(axis, rank, op)
    };
    let a_axes: Vec<usize> = axes
        .0
        .iter()
        .map(|a| norm_axis(*a, a_shape.len(), "np.tensordot"))
        .collect();
    let b_axes: Vec<usize> = axes
        .1
        .iter()
        .map(|a| norm_axis(*a, b_shape.len(), "np.tensordot"))
        .collect();
    if a_axes.len() != b_axes.len() {
        panic!("np.tensordot: axes lists must have same length");
    }
    for (aa, ba) in a_axes.iter().zip(b_axes.iter()) {
        if a_shape[*aa] != b_shape[*ba] {
            panic!("np.tensordot: contracted dimensions must match");
        }
    }
    let a_keep: Vec<usize> = (0..a_shape.len())
        .filter(|ax| !a_axes.contains(ax))
        .collect();
    let b_keep: Vec<usize> = (0..b_shape.len())
        .filter(|ax| !b_axes.contains(ax))
        .collect();
    let mut out_shape: Vec<usize> = a_keep.iter().map(|ax| a_shape[*ax]).collect();
    out_shape.extend(b_keep.iter().map(|ax| b_shape[*ax]));
    let contract_shape: Vec<usize> = a_axes.iter().map(|ax| a_shape[*ax]).collect();
    let out_total: usize = out_shape.iter().product();
    let contract_total: usize = contract_shape.iter().product();
    let out_strides = crate::helpers::shape_arith::row_major_strides(&out_shape);
    let contract_strides = crate::helpers::shape_arith::row_major_strides(&contract_shape);
    let a_strides = crate::helpers::shape_arith::row_major_strides(&a_shape);
    let b_strides = crate::helpers::shape_arith::row_major_strides(&b_shape);
    let a_cells = static_cells(b, &args[0], "np.tensordot");
    let b_cells = static_cells(b, &args[1], "np.tensordot");
    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let out_coords =
            crate::helpers::shape_arith::decode_coords(out_idx, &out_shape, &out_strides);
        let mut a_base = vec![0usize; a_shape.len()];
        let mut b_base = vec![0usize; b_shape.len()];
        let mut cursor = 0usize;
        for ax in &a_keep {
            a_base[*ax] = out_coords[cursor];
            cursor += 1;
        }
        for ax in &b_keep {
            b_base[*ax] = out_coords[cursor];
            cursor += 1;
        }
        let mut lhs = Vec::with_capacity(contract_total);
        let mut rhs = Vec::with_capacity(contract_total);
        for cidx in 0..contract_total {
            let ccoords = if contract_shape.is_empty() {
                Vec::new()
            } else {
                crate::helpers::shape_arith::decode_coords(cidx, &contract_shape, &contract_strides)
            };
            let mut ac = a_base.clone();
            let mut bc = b_base.clone();
            for (k, coord) in ccoords.iter().enumerate() {
                ac[a_axes[k]] = *coord;
                bc[b_axes[k]] = *coord;
            }
            lhs.push(a_cells[crate::helpers::shape_arith::encode_coords(&ac, &a_strides)].clone());
            rhs.push(b_cells[crate::helpers::shape_arith::encode_coords(&bc, &b_strides)].clone());
        }
        out.push(product_sum(b, &lhs, &rhs));
    }
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        out,
        out_shape,
        NumberType::Float,
    )
}

fn np_diag(b: &mut crate::builder::IRBuilder, args: &[Value]) -> Value {
    assert!(!args.is_empty(), "np.diag: requires an input array");
    let k = args.get(1).and_then(|v| v.int_val()).unwrap_or(0);
    if k != 0 {
        panic!("np.diag: nonzero k is not yet supported");
    }
    let arr = &args[0];
    if matches!(arr, Value::DynamicNDArray(_)) {
        panic!("np.diag is not yet supported for bounded-dynamic arrays; output shape depends on runtime metadata");
    }
    let (dtype, shape, cells) = match arr {
        Value::StaticArray { dtype, shape, .. } => (
            *dtype,
            shape.clone(),
            crate::helpers::static_array::elementwise::payload_cells(b, arr),
        ),
        Value::List(_) | Value::Tuple(_) => {
            let promoted = crate::helpers::static_array::base::to_static_array(b, arr)
                .unwrap_or_else(|| panic!("np.diag: input must be a numeric array"));
            let Value::StaticArray { dtype, shape, .. } = &promoted else {
                unreachable!()
            };
            (
                *dtype,
                shape.clone(),
                crate::helpers::static_array::elementwise::payload_cells(b, &promoted),
            )
        }
        _ => panic!("np.diag: input must be a 1-D or 2-D array"),
    };
    match shape.as_slice() {
        [n] => {
            let total = n * n;
            let zero = match dtype {
                NumberType::Float => b.ir_constant_float(0.0),
                NumberType::Integer => b.ir_constant_int(0),
                NumberType::Complex => crate::helpers::value_ops::pack_complex_value(
                    b.ir_constant_float(0.0),
                    b.ir_constant_float(0.0),
                ),
            };
            let mut out = vec![zero; total];
            for i in 0..*n {
                out[i * n + i] = cells[i].clone();
            }
            if dtype == NumberType::Complex {
                let mut reals = Vec::with_capacity(total);
                let mut imags = Vec::with_capacity(total);
                for cell in out {
                    match cell {
                        Value::Complex { real, imag } => {
                            reals.push(Value::Float(real));
                            imags.push(Value::Float(imag));
                        }
                        _ => unreachable!("Complex diag cell expected"),
                    }
                }
                crate::helpers::static_array::base::build_static_array_from_flat_complex(
                    b,
                    reals,
                    imags,
                    vec![*n, *n],
                )
            } else {
                crate::helpers::static_array::base::build_static_array_from_flat(
                    b,
                    out,
                    vec![*n, *n],
                    dtype,
                )
            }
        }
        [rows, cols] => {
            let n = (*rows).min(*cols);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(cells[i * cols + i].clone());
            }
            if dtype == NumberType::Complex {
                let mut reals = Vec::with_capacity(n);
                let mut imags = Vec::with_capacity(n);
                for cell in out {
                    match cell {
                        Value::Complex { real, imag } => {
                            reals.push(Value::Float(real));
                            imags.push(Value::Float(imag));
                        }
                        _ => unreachable!("Complex diag cell expected"),
                    }
                }
                crate::helpers::static_array::base::build_static_array_from_flat_complex(
                    b,
                    reals,
                    imags,
                    vec![n],
                )
            } else {
                crate::helpers::static_array::base::build_static_array_from_flat(
                    b,
                    out,
                    vec![n],
                    dtype,
                )
            }
        }
        _ => panic!("np.diag: input must be 1-D or 2-D"),
    }
}

fn np_indices(b: &mut crate::builder::IRBuilder, args: &[Value]) -> Value {
    assert!(!args.is_empty(), "np.indices: requires dimensions");
    let dims_value = &args[0];
    let dims: Vec<usize> = match dims_value {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                let n = v.int_val().unwrap_or_else(|| {
                    panic!("np.indices: dimensions must be compile-time integers")
                });
                assert!(n >= 0, "np.indices: dimensions must be nonnegative");
                n as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n = dims_value
                .int_val()
                .unwrap_or_else(|| panic!("np.indices: dimension must be compile-time integer"));
            assert!(n >= 0, "np.indices: dimension must be nonnegative");
            vec![n as usize]
        }
        _ => panic!("np.indices: dimensions must be a tuple/list of integers"),
    };
    if dims.is_empty() {
        panic!("np.indices: dimensions must not be empty");
    }
    let dense = args
        .get(2)
        .and_then(|v| v.bool_val())
        .or_else(|| {
            if let Some(Value::Boolean(_)) = args.get(1) {
                args[1].bool_val()
            } else {
                None
            }
        })
        .unwrap_or(true);
    if !dense {
        panic!("np.indices(sparse=True) is not yet supported");
    }
    let total: usize = dims.iter().product();
    let mut out_shape = Vec::with_capacity(dims.len() + 1);
    out_shape.push(dims.len());
    out_shape.extend(dims.iter().copied());
    let strides = crate::helpers::shape_arith::row_major_strides(&dims);
    let mut out = Vec::with_capacity(dims.len() * total);
    for axis in 0..dims.len() {
        for flat in 0..total {
            let coord = if total == 0 {
                0
            } else {
                crate::helpers::shape_arith::decode_coords(flat, &dims, &strides)[axis]
            };
            out.push(b.ir_constant_int(coord as i64));
        }
    }
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        out,
        out_shape,
        NumberType::Integer,
    )
}

/// Check if the first arg is a list/tuple containing DynamicNDArrays.
fn has_dynamic_array_in_list(args: &[Value]) -> bool {
    match args.first() {
        Some(Value::List(cd)) | Some(Value::Tuple(cd)) => cd
            .values
            .iter()
            .any(|v| matches!(v, Value::DynamicNDArray(_))),
        _ => false,
    }
}

/// P4c: helper to detect StaticArray inputs in a list/tuple — used to
/// decide whether to route concatenate/stack/etc. through the native
/// path before falling back to the legacy nested-List handlers.
fn list_contains_static_array(arg: &Value) -> bool {
    match arg {
        Value::List(cd) | Value::Tuple(cd) => cd
            .values
            .iter()
            .any(|v| matches!(v, Value::StaticArray { .. })),
        _ => false,
    }
}

impl IRGenerator {
    pub(crate) fn try_numpy_func(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let visited_args = ctx.args;
        let _visited_kwargs = ctx.kwargs;
        let visited_args_orig = ctx.args_orig;
        let visited_kwargs_orig = ctx.kwargs_orig;
        let target = ctx.target?;
        let member = ctx.member;

        let v = match (target, member) {
            // ── np.* (numpy-like operations) ───────────────────────────
            // numpy dtype aliases — collapse width into ZinniaType (Float / Integer / Boolean)
            (
                "np",
                "float16" | "float32" | "float64" | "double" | "single" | "half" | "longdouble",
            ) => Value::Class(ZinniaType::Float),
            (
                "np",
                "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
                | "intp" | "uintp" | "long" | "byte" | "short" | "intc" | "uintc" | "ubyte"
                | "ushort" | "longlong" | "ulonglong",
            ) => Value::Class(ZinniaType::Integer),
            ("np", "bool_") => Value::Class(ZinniaType::Boolean),
            (
                "np",
                "complex64" | "complex128" | "complex256" | "csingle" | "cdouble" | "clongdouble",
            ) => Value::Class(ZinniaType::Complex),
            // Complex helpers
            ("np", "conj" | "conjugate") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { real, imag } => {
                        // (a + bi) -> (a - bi)
                        let zero = self.builder.ir_constant_float(0.0);
                        let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag));
                        let ni = match neg_imag {
                            Value::Float(s) => s,
                            _ => unreachable!(),
                        };
                        Value::Complex { real, imag: ni }
                    }
                    // P5a: Complex StaticArray → fresh imag segment with negated values; share real segment.
                    Value::StaticArray {
                        dtype: crate::types::NumberType::Complex,
                        ..
                    } => crate::helpers::static_array::complex::np_conj_static_array(
                        &mut self.builder,
                        &v,
                    ),
                    Value::DynamicNDArray(d) => {
                        crate::ops::dyn_ndarray::metadata::dyn_complex_conjugate(
                            &mut self.builder,
                            &d,
                        )
                    }
                    // For real inputs, conj is the identity.
                    other => other,
                }
            }
            ("np", "real") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { real, .. } => Value::Float(real),
                    Value::StaticArray {
                        dtype: crate::types::NumberType::Complex,
                        ..
                    } => crate::helpers::static_array::complex::np_real_static_array(
                        &mut self.builder,
                        &v,
                    ),
                    Value::DynamicNDArray(d) => {
                        crate::ops::dyn_ndarray::metadata::dyn_complex_real(
                            &mut self.builder,
                            &d,
                        )
                    }
                    // For real inputs, real is the identity.
                    other => other,
                }
            }
            ("np", "imag") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { imag, .. } => Value::Float(imag),
                    Value::StaticArray {
                        dtype: crate::types::NumberType::Complex,
                        ..
                    } => crate::helpers::static_array::complex::np_imag_static_array(
                        &mut self.builder,
                        &v,
                    ),
                    Value::DynamicNDArray(d) => {
                        crate::ops::dyn_ndarray::metadata::dyn_complex_imag(
                            &mut self.builder,
                            &d,
                        )
                    }
                    // For real inputs, imag is zero of same shape.
                    Value::StaticArray { shape, .. } => {
                        let total: usize = shape.iter().product();
                        let zero = self.builder.ir_constant_float(0.0);
                        crate::helpers::static_array::base::build_static_array_from_flat(
                            &mut self.builder,
                            vec![zero; total],
                            shape,
                            crate::types::NumberType::Float,
                        )
                    }
                    Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => {
                        self.builder.ir_constant_float(0.0)
                    }
                    other => other,
                }
            }
            // numpy / math mathematical constants
            ("np", "pi") | ("math", "pi") => self.builder.ir_constant_float(std::f64::consts::PI),
            ("np", "e") | ("math", "e") => self.builder.ir_constant_float(std::f64::consts::E),
            ("np", "asarray" | "array") => {
                // np.asarray(x) — pass through if already a composite
                if !visited_args.is_empty() {
                    let val = &visited_args[0];
                    // Validate that all sub-elements have consistent shapes
                    if let Value::List(data) | Value::Tuple(data) = val {
                        if data.values.len() > 1 {
                            let first_is_composite =
                                matches!(&data.values[0], Value::List(_) | Value::Tuple(_));
                            for v in &data.values[1..] {
                                let is_composite = matches!(v, Value::List(_) | Value::Tuple(_));
                                if is_composite != first_is_composite {
                                    panic!("To convert to NDArray, all sub-lists should be of the same shape");
                                }
                            }
                        }
                    }
                    // Handle dtype kwarg for type casting
                    let cast_val = match _visited_kwargs.get("dtype") {
                        Some(Value::Class(ZinniaType::Boolean)) => self.truth_composite(val),
                        Some(dtype) => {
                            let to_float = matches!(dtype, Value::Class(ZinniaType::Float));
                            self.cast_composite(val, to_float)
                        }
                        None => val.clone(),
                    };
                    // P1 segarr-foundation: numeric Python list/tuple inputs
                    // become segment-backed `Value::StaticArray`. Falls back
                    // to passing through for non-numeric / heterogeneous data.
                    if let Some(sa) = crate::helpers::static_array::base::to_static_array(
                        &mut self.builder,
                        &cast_val,
                    ) {
                        sa
                    } else {
                        cast_val
                    }
                } else {
                    Value::None
                }
            }
            ("np", "zeros") => crate::ops::static_ndarray_ops::np_fill(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                0,
            ),
            ("np", "ones") => crate::ops::static_ndarray_ops::np_fill(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                1,
            ),
            ("np", "empty" | "ndarray") => crate::ops::static_ndarray_ops::np_fill(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                0,
            ),
            ("np", "full") => crate::ops::static_ndarray_ops::np_full(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "zeros_like") => crate::ops::static_ndarray_ops::np_fill_like(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                0,
            ),
            ("np", "ones_like") => crate::ops::static_ndarray_ops::np_fill_like(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                1,
            ),
            ("np", "empty_like") => crate::ops::static_ndarray_ops::np_fill_like(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
                0,
            ),
            ("np", "full_like") => crate::ops::static_ndarray_ops::np_full_like(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "identity" | "eye") => {
                crate::ops::static_ndarray_ops::np_identity(&mut self.builder, visited_args)
            }
            ("np", method @ ("argmax" | "argmin")) => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &val_orig,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return Some(out);
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.get(1));
                crate::helpers::array_ops::argmax_argmin(
                    &mut self.builder,
                    &val,
                    axis_arg,
                    method == "argmax",
                )
            }
            ("np", method @ ("sort" | "argsort")) => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                if let Some(out) = crate::helpers::static_array::sorting::try_apply_sort(
                    &mut self.builder,
                    &val_orig,
                    axis_arg_orig,
                    method == "argsort",
                ) {
                    return Some(out);
                }
                panic!(
                    "np.{}: only StaticArray inputs are supported; dynamic sort/argsort is not yet implemented",
                    method
                );
            }
            ("np", "inner") => {
                let lhs = &visited_args_orig[0]; let rhs = &visited_args_orig[1];
                if crate::helpers::composite::get_composite_shape(lhs).is_empty() || crate::helpers::composite::get_composite_shape(rhs).is_empty() {
                    crate::helpers::value_ops::apply_binary_op(&mut self.builder,"mul",lhs,rhs)
                } else {
                    let minus = self.builder.ir_constant_int(-1);
                    let axis = Value::Tuple(CompositeData::new(vec![ZinniaType::Integer],vec![minus]));
                    let axes = Value::Tuple(CompositeData::new(vec![axis.zinnia_type(),axis.zinnia_type()],vec![axis.clone(),axis]));
                    let mut kw=HashMap::new();kw.insert("axes".to_string(),axes);
                    np_tensordot_ds1000(&mut self.builder,visited_args_orig,&kw)
                }
            }
            ("np", "matmul") => {
                crate::helpers::value_ops::apply_binary_op(&mut self.builder, "mat_mul", &visited_args_orig[0], &visited_args_orig[1])
            }
            ("np", "dot") => {
                let lhs = visited_args.first().cloned().unwrap_or(Value::None);
                let rhs = visited_args.get(1).cloned().unwrap_or(Value::None);
                if matches!(lhs, Value::DynamicNDArray(_)) || matches!(rhs, Value::DynamicNDArray(_)) {
                    panic!(
                        "np.{} is not yet supported for bounded-dynamic arrays; use static arrays or keep matmul outside the bounded-dynamic surface",
                        member
                    );
                }
                crate::ops::static_ndarray_ops::matmul(&mut self.builder, &lhs, &rhs)
            }
            ("np", "square") => {
                crate::ops::static_ndarray_ops::np_square(&mut self.builder, visited_args_orig)
            }
            ("np", "deg2rad" | "radians") => {
                let x = visited_args.first().cloned().unwrap_or(Value::None);
                let scale = self
                    .builder
                    .ir_constant_float(std::f64::consts::PI / 180.0);
                crate::helpers::value_ops::apply_binary_op(&mut self.builder, "mul", &x, &scale)
            }
            ("np", "rad2deg" | "degrees") => {
                let x = visited_args.first().cloned().unwrap_or(Value::None);
                let scale = self
                    .builder
                    .ir_constant_float(180.0 / std::f64::consts::PI);
                crate::helpers::value_ops::apply_binary_op(&mut self.builder, "mul", &x, &scale)
            }
            ("np", "diff") => {
                crate::ops::static_ndarray_ops::np_diff(&mut self.builder, visited_args)
            }
            ("np", "outer") => {
                crate::ops::static_ndarray_ops::np_outer(&mut self.builder, visited_args)
            }
            ("np", "add_outer") => {
                crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "add")
            }
            ("np", "subtract_outer") => {
                crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "sub")
            }
            ("np", "multiply_outer") => {
                crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "mul")
            }
            ("np", "divide_outer") => {
                crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "div")
            }
            ("np", "arange") => crate::ops::static_ndarray_ops::np_arange(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "linspace") => crate::ops::static_ndarray_ops::np_linspace(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "allclose") => crate::ops::static_ndarray_ops::np_allclose(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "isclose") => {
                // Use the un-materialised args: ndarray operands must keep
                // StaticArray representation so the comparison chain stays
                // elementwise (a List pair would collapse to one Boolean
                // under Python sequence equality).
                np_isclose(&mut self.builder, visited_args_orig, _visited_kwargs)
            }
            ("np", "array_equal") => np_array_equal(&mut self.builder, visited_args),
            ("np", "concatenate") => {
                if has_dynamic_array_in_list(visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                } else {
                    // P4c: native StaticArray dispatch — uses orig args (un-converted).
                    let raw_axis = visited_kwargs_orig
                        .get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .and_then(|v| v.int_val())
                        .unwrap_or(0);
                    if let Some(arrays_arg) = visited_args_orig.first() {
                        if list_contains_static_array(arrays_arg) {
                            if let Some(out) =
                                crate::helpers::static_array::shape::try_apply_concatenate(
                                    &mut self.builder,
                                    arrays_arg,
                                    raw_axis,
                                )
                            {
                                return Some(out);
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_concatenate(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                }
            }
            ("np", "stack") => {
                if has_dynamic_array_in_list(visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_stack(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                } else {
                    // P4c: native StaticArray dispatch.
                    let raw_axis = visited_kwargs_orig
                        .get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .and_then(|v| v.int_val())
                        .unwrap_or(0);
                    if let Some(arrays_arg) = visited_args_orig.first() {
                        if list_contains_static_array(arrays_arg) {
                            if let Some(out) = crate::helpers::static_array::shape::try_apply_stack(
                                &mut self.builder,
                                arrays_arg,
                                raw_axis,
                            ) {
                                return Some(out);
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_stack(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                }
            }
            ("np", "vstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array::shape::try_apply_vstack(
                            &mut self.builder,
                            arrays_arg,
                        ) {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_vstack(&mut self.builder, visited_args)
            }
            ("np", "row_stack") => {
                crate::ops::static_ndarray_ops::np_row_stack(&mut self.builder, visited_args)
            }
            ("np", "hstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array::shape::try_apply_hstack(
                            &mut self.builder,
                            arrays_arg,
                        ) {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_hstack(&mut self.builder, visited_args)
            }
            ("np", "dstack") => {
                crate::ops::static_ndarray_ops::np_dstack(&mut self.builder, visited_args)
            }
            ("np", "column_stack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) =
                            crate::helpers::static_array::shape::try_apply_column_stack(
                                &mut self.builder,
                                arrays_arg,
                            )
                        {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_column_stack(&mut self.builder, visited_args)
            }
            ("np", "swapaxes") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_swapaxes(
                    &mut self.builder,
                    &val,
                    &visited_args[1..],
                )
            }
            ("np", "moveaxis") => {
                // P4c: native StaticArray dispatch via array_ops::moveaxis fast-path,
                // using the un-converted orig args.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    return Some(crate::helpers::array_ops::moveaxis(
                        &mut self.builder,
                        &val_orig,
                        &visited_args_orig[1..],
                    ));
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_moveaxis(
                    &mut self.builder,
                    &val,
                    &visited_args[1..],
                )
            }
            ("np", "transpose") => {
                // P4c: native StaticArray dispatch.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_transpose(
                        &mut self.builder,
                        &val_orig,
                        &visited_args_orig[1..],
                    ) {
                        return Some(out);
                    }
                }
                if matches!(val_orig, Value::DynamicNDArray(_)) {
                    return Some(crate::helpers::array_ops::transpose(
                        &mut self.builder,
                        &val_orig,
                        &visited_args_orig[1..],
                    ));
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::ndarray::ndarray_transpose(
                    &mut self.builder,
                    &val,
                    &visited_args[1..],
                )
            }
            ("np", "flip") => crate::ops::static_ndarray_ops::np_flip(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "flipud") => {
                crate::ops::static_ndarray_ops::np_flipud(&mut self.builder, visited_args)
            }
            ("np", "fliplr") => {
                crate::ops::static_ndarray_ops::np_fliplr(&mut self.builder, visited_args)
            }
            ("np", "rot90") => crate::ops::static_ndarray_ops::np_rot90(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "squeeze") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_kwargs_orig
                        .get("axis")
                        .or_else(|| visited_args_orig.get(1));
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_squeeze(
                        &mut self.builder,
                        &val_orig,
                        axis_arg,
                    ) {
                        return Some(out);
                    }
                }
                if let Value::DynamicNDArray(d) = &val_orig {
                    let axis_args: Vec<Value> = visited_kwargs_orig
                        .get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .cloned()
                        .into_iter()
                        .collect();
                    return Some(crate::ops::dyn_ndarray::reshape::dyn_squeeze(
                        &mut self.builder,
                        d,
                        &axis_args,
                    ));
                }
                crate::ops::static_ndarray_ops::np_squeeze(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("np", "expand_dims") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_args_orig.get(1);
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_expand_dims(
                        &mut self.builder,
                        &val_orig,
                        axis_arg,
                    ) {
                        return Some(out);
                    }
                }
                if let Value::DynamicNDArray(d) = &val_orig {
                    let axis_args: Vec<Value> = visited_kwargs_orig
                        .get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .cloned()
                        .into_iter()
                        .collect();
                    return Some(crate::ops::dyn_ndarray::reshape::dyn_expand_dims(
                        &mut self.builder,
                        d,
                        &axis_args,
                    ));
                }
                crate::ops::static_ndarray_ops::np_expand_dims(&mut self.builder, visited_args)
            }
            ("np", "broadcast_to") => {
                crate::ops::static_ndarray_ops::np_broadcast_to(&mut self.builder, visited_args)
            }
            ("np", "atleast_1d") => {
                crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 1)
            }
            ("np", "atleast_2d") => {
                crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 2)
            }
            ("np", "atleast_3d") => {
                crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 3)
            }
            ("np", "tile") => {
                crate::ops::static_ndarray_ops::np_tile(&mut self.builder, visited_args)
            }
            ("np", "take") => {
                let arr_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let indices_orig = visited_args_orig.get(1).cloned().unwrap_or(Value::None);
                let axis_raw_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(2));
                if let Some(out) = crate::helpers::static_array::sorting::try_apply_take(
                    &mut self.builder,
                    &arr_orig,
                    &indices_orig,
                    axis_raw_orig,
                ) {
                    return Some(out);
                }
                // Legacy composite path: default-axis case on a 1-D source,
                // or explicit axis=0. Multi-D flatten-then-take and arbitrary
                // axes are handled by the StaticArray path above or rejected.
                let arr = visited_args.first().cloned().unwrap_or(Value::None);
                let indices = visited_args.get(1).cloned().unwrap_or(Value::None);
                let axis_raw = _visited_kwargs.get("axis").or_else(|| visited_args.get(2));
                let axis_is_default = match axis_raw {
                    None => true,
                    Some(Value::None) => true,
                    _ => false,
                };
                let axis_val = axis_raw.and_then(|v| v.int_val()).unwrap_or(0);
                let data = match &arr {
                    Value::List(d) | Value::Tuple(d) => d.clone(),
                    _ => panic!("np.take: first argument must be a static-shape array"),
                };
                // Default-axis path on a 1-D source matches axis=0; for
                // multi-D `axis=None` we'd need to flatten first.
                let is_1d = data
                    .values
                    .iter()
                    .all(|v| !matches!(v, Value::List(_) | Value::Tuple(_)));
                if !axis_is_default && axis_val != 0 {
                    panic!("np.take with axis != 0 is not yet implemented");
                }
                if axis_is_default && !is_1d {
                    panic!("np.take on multi-dimensional arrays with axis=None (flatten) is not yet implemented");
                }
                let out = match crate::helpers::ndarray::fancy_index_static(&data, &indices) {
                    Ok(v) => v,
                    Err(msg) => panic!("np.take: {}", msg),
                };
                // Group 5c (E-axis): content fact relay.
                if let (Some(in_vid), Some(out_vid)) = (arr.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(
                        &mut self.builder,
                        in_vid,
                        out_vid,
                    );
                }
                out
            }
            ("np", "split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(
                        &mut self.builder,
                        &data,
                        &visited_args[1..],
                        _visited_kwargs,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_split(
                        &mut self.builder,
                        visited_args_orig,
                        visited_kwargs_orig,
                    )
                }
            }
            ("np", "array_split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_array_split(
                        &mut self.builder,
                        &data,
                        &visited_args[1..],
                        _visited_kwargs,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_array_split(
                        &mut self.builder,
                        visited_args_orig,
                        visited_kwargs_orig,
                    )
                }
            }
            ("np", "hsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let ndim = data.meta.logical_shape.len();
                    let ax = if ndim == 1 { 0i64 } else { 1i64 };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert(
                        "axis".to_string(),
                        Value::Integer(ScalarValue::new(Some(ax), None)),
                    );
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(
                        &mut self.builder,
                        &data,
                        &visited_args[1..],
                        &kw,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_hsplit(&mut self.builder, visited_args_orig)
                }
            }
            ("np", "vsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert(
                        "axis".to_string(),
                        Value::Integer(ScalarValue::new(Some(0), None)),
                    );
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(
                        &mut self.builder,
                        &data,
                        &visited_args[1..],
                        &kw,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_vsplit(&mut self.builder, visited_args_orig)
                }
            }
            ("np", "dsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert(
                        "axis".to_string(),
                        Value::Integer(ScalarValue::new(Some(2), None)),
                    );
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(
                        &mut self.builder,
                        &data,
                        &visited_args[1..],
                        &kw,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_dsplit(&mut self.builder, visited_args_orig)
                }
            }
            ("np", "floor") => {
                match visited_args.first() {
                    Some(Value::DynamicNDArray(_)) => panic!(
                        "np.floor is not yet supported for bounded-dynamic arrays; dynamic Float rounding needs active-lane-aware fixed-point rounding constraints"
                    ),
                    _ => crate::ops::static_ndarray_ops::np_floor(&mut self.builder, visited_args),
                }
            }
            ("np", "ceil") => {
                match visited_args.first() {
                    Some(Value::DynamicNDArray(_)) => panic!(
                        "np.ceil is not yet supported for bounded-dynamic arrays; dynamic Float rounding needs active-lane-aware fixed-point rounding constraints"
                    ),
                    _ => crate::ops::static_ndarray_ops::np_ceil(&mut self.builder, visited_args),
                }
            }
            ("np", "trunc") => {
                match visited_args.first() {
                    Some(Value::DynamicNDArray(_)) => panic!(
                        "np.trunc is not yet supported for bounded-dynamic arrays; dynamic Float rounding needs active-lane-aware fixed-point rounding constraints"
                    ),
                    _ => crate::ops::static_ndarray_ops::np_trunc(&mut self.builder, visited_args),
                }
            }
            ("np", "round") => {
                match visited_args.first() {
                    Some(Value::DynamicNDArray(_)) => panic!(
                        "np.round is not yet supported for bounded-dynamic arrays; dynamic Float rounding needs active-lane-aware fixed-point rounding constraints"
                    ),
                    _ => crate::ops::static_ndarray_ops::np_round(&mut self.builder, visited_args),
                }
            }
            ("np", "reciprocal") => {
                match visited_args.first() {
                    Some(Value::DynamicNDArray(_)) => panic!(
                        "np.reciprocal is not yet supported for bounded-dynamic arrays because each active element needs a nonzero proof obligation"
                    ),
                    _ => crate::ops::static_ndarray_ops::np_reciprocal(&mut self.builder, visited_args),
                }
            }
            ("np", "where") => {
                // 1-arg `np.where(cond)` forwards the boolean mask (for the
                // `a[np.where(cond)] = v` pattern). Use the ORIGINAL,
                // un-materialised arg: the legacy-materialised List view
                // loses the StaticArray representation the boolean-mask
                // write dispatch recognizes, silently degrading the mask
                // to a scalar dynamic index (binding-audit case 0490).
                if visited_args_orig.len() == 1 {
                    crate::ops::static_ndarray_ops::np_where(&mut self.builder, visited_args_orig)
                } else {
                    crate::ops::static_ndarray_ops::np_where(&mut self.builder, visited_args)
                }
            }
            ("np", "diag") => np_diag(&mut self.builder, visited_args),
            ("np", "indices") => np_indices(&mut self.builder, visited_args),
            ("np", "unravel_index") => crate::ops::static_ndarray_ops::np_unravel_index(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "ravel_multi_index") => crate::ops::static_ndarray_ops::np_ravel_multi_index(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "take_along_axis") => crate::ops::static_ndarray_ops::np_take_along_axis(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "delete") => crate::ops::static_ndarray_ops::np_delete(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "insert") => crate::ops::static_ndarray_ops::np_insert(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "pad") => crate::ops::static_ndarray_ops::np_pad(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "argwhere") => {
                crate::ops::static_ndarray_ops::np_argwhere(&mut self.builder, visited_args)
            }
            ("np", "nonzero") => {
                crate::ops::static_ndarray_ops::np_nonzero(&mut self.builder, visited_args)
            }
            ("np", "flatnonzero") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let flat = if matches!(val_orig, Value::StaticArray { .. }) {
                    crate::helpers::static_array::shape::try_apply_flatten(
                        &mut self.builder,
                        &val_orig,
                    )
                    .unwrap_or(val_orig)
                } else {
                    visited_args.first().cloned().unwrap_or(Value::None)
                };
                let nz = crate::ops::static_ndarray_ops::np_nonzero(&mut self.builder, &[flat]);
                match nz {
                    Value::Tuple(d) if d.values.len() == 1 => d.values[0].clone(),
                    _ => panic!("np.flatnonzero: expected rank-1 nonzero output"),
                }
            }
            ("np", "in1d") => {
                let ar1 = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let ar2 = visited_args_orig.get(1).cloned().unwrap_or(Value::None);
                let ar1_flat = if matches!(ar1, Value::StaticArray { .. }) {
                    crate::helpers::static_array::shape::try_apply_flatten(&mut self.builder, &ar1)
                        .unwrap_or(ar1)
                } else {
                    ar1
                };
                let ar2_flat = if matches!(ar2, Value::StaticArray { .. }) {
                    crate::helpers::static_array::shape::try_apply_flatten(&mut self.builder, &ar2)
                        .unwrap_or(ar2)
                } else {
                    ar2
                };
                let (shape, ar1_cells) = match &ar1_flat {
                    Value::StaticArray { shape, .. } => (
                        shape.clone(),
                        crate::helpers::static_array::elementwise::payload_cells(
                            &mut self.builder,
                            &ar1_flat,
                        ),
                    ),
                    _ => panic!("np.in1d: ar1 must be a StaticArray in this slice"),
                };
                let ar2_cells = match &ar2_flat {
                    Value::StaticArray { .. } => {
                        crate::helpers::static_array::elementwise::payload_cells(
                            &mut self.builder,
                            &ar2_flat,
                        )
                    }
                    Value::List(d) | Value::Tuple(d) => d.values.clone(),
                    _ => panic!("np.in1d: ar2 must be a static array/list in this slice"),
                };
                let mut out = Vec::with_capacity(ar1_cells.len());
                for lhs in ar1_cells {
                    let mut found = self.builder.ir_constant_bool(false);
                    for rhs in &ar2_cells {
                        let eq = crate::helpers::value_ops::apply_binary_op(
                            &mut self.builder,
                            "eq",
                            &lhs,
                            rhs,
                        );
                        let eq_bool =
                            crate::helpers::value_ops::to_scalar_bool(&mut self.builder, &eq);
                        found = self.builder.ir_logical_or(&found, &eq_bool);
                    }
                    out.push(found);
                }
                crate::helpers::static_array::base::build_static_array_from_flat(
                    &mut self.builder,
                    out,
                    shape,
                    crate::types::NumberType::Integer,
                )
            }
            ("np", "bincount") => {
                crate::ops::static_ndarray_ops::np_bincount(
                    &mut self.builder,
                    visited_args_orig,
                    visited_kwargs_orig,
                )
            }
            ("np", "histogram") => crate::ops::static_ndarray_ops::np_histogram(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "percentile") => np_percentile_ds1000(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "correlate") => np_correlate_ds1000(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "corrcoef") => np_corrcoef_ds1000(&mut self.builder, visited_args_orig),
            ("np", "einsum") => {
                ensure_no_kwargs("np.einsum", visited_kwargs_orig, &["optimize"]);
                crate::ops::static_ndarray_ops::einsum::einsum(&mut self.builder, visited_args_orig)
            }
            ("np", "tensordot") => np_tensordot_ds1000(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "unique") => crate::ops::static_ndarray_ops::np_unique(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "cov") => crate::ops::static_ndarray_ops::np_cov(
                &mut self.builder,
                visited_args_orig,
                visited_kwargs_orig,
            ),
            ("np", "copyto") => {
                let [target, source] = visited_args_orig else {
                    panic!("np.copyto(dst, src, where=...) takes two positional arguments");
                };
                let truth = self.builder.ir_constant_bool(true);
                let mask = visited_kwargs_orig.get("where").unwrap_or(&truth);
                let written = super::array_compat::write_output(&mut self.builder, source, target, Some(mask));
                self.note_array_write(&written);
                Value::None
            }
            ("np", "putmask") => {
                let [target, mask, values] = visited_args_orig else {
                    panic!("np.putmask(a, mask, values) takes three positional arguments");
                };
                let written = super::array_compat::putmask(&mut self.builder, target, mask, values);
                self.note_array_write(&written);
                Value::None
            }
            ("np", "select") => {
                let conditions = match &visited_args[0] { Value::List(d)|Value::Tuple(d)=>&d.values,_=>panic!("select: expected condition list") };
                let choices = match &visited_args[1] { Value::List(d)|Value::Tuple(d)=>&d.values,_=>panic!("select: expected choice list") };
                assert_eq!(conditions.len(),choices.len(),"select: lists must have equal length");
                assert!(!conditions.is_empty(),"select: empty condition list");
                let mut result = _visited_kwargs.get("default").or_else(||visited_args.get(2)).cloned().unwrap_or_else(||self.builder.ir_constant_int(0));
                for (cond,choice) in conditions.iter().zip(choices).rev() {
                    result = crate::ops::static_ndarray_ops::np_where(&mut self.builder,&[cond.clone(),choice.clone(),result]);
                }
                crate::helpers::static_array::base::to_static_array(&mut self.builder,&result).unwrap_or(result)
            }
            ("np", "median") => super::array_compat::median(&mut self.builder,visited_args_orig,visited_kwargs_orig),
            ("np", "abs" | "absolute") => {
                np_abs_value(&mut self.builder, &visited_args_orig[0])
            }
            ("np", "clip") => {
                super::array_compat::clip(&mut self.builder, visited_args_orig, visited_kwargs_orig)
            }
            ("np", "copy") => {
                ensure_no_kwargs("np.copy", visited_kwargs_orig, &["order", "subok"]);
                let input = visited_args_orig.first().expect("np.copy: requires an array");
                super::array_compat::copy_array(&mut self.builder, input)
            }
            ("np", "mean") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_reduce(
                    &mut self.builder,
                    "mean",
                    &val_orig,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return Some(out);
                }
                crate::ops::static_ndarray_ops::np_mean(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("np", op @ ("var" | "std")) => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                let ddof_arg_orig = visited_kwargs_orig
                    .get("ddof")
                    .or_else(|| visited_args_orig.get(4));
                let unsupported_dtype = visited_kwargs_orig
                    .get("dtype")
                    .or_else(|| visited_args_orig.get(2));
                if !matches!(unsupported_dtype, None | Some(Value::None)) {
                    panic!(
                        "{}: dtype argument is deferred for native StaticArray stats",
                        op
                    );
                }
                let unsupported_out = visited_kwargs_orig
                    .get("out")
                    .or_else(|| visited_args_orig.get(3));
                if !matches!(unsupported_out, None | Some(Value::None)) {
                    panic!(
                        "{}: out argument is deferred for native StaticArray stats",
                        op
                    );
                }
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_var_std(
                    &mut self.builder,
                    op,
                    &val_orig,
                    axis_arg_orig,
                    ddof_arg_orig,
                    keepdims,
                ) {
                    return Some(out);
                }
                if op == "var" {
                    crate::ops::static_ndarray_ops::np_var(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_std(
                        &mut self.builder,
                        visited_args,
                        _visited_kwargs,
                    )
                }
            }
            ("np", "cumsum") => crate::ops::static_ndarray_ops::np_cumsum(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "cumprod") => crate::ops::static_ndarray_ops::np_cumprod(
                &mut self.builder,
                visited_args,
                _visited_kwargs,
            ),
            ("np", "promote_to_dynamic") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::promote::promote_static_to_dynamic(&mut self.builder, &val)
            }
            ("np", "block") => {
                // Block depth must be measured from the *AST*, not the
                // visited Value: after Python list literals have been turned
                // into `Value::List`, they are indistinguishable from
                // ndarray axes. Walk the leftmost path of the original AST,
                // counting only Python list/tuple literals.
                fn ast_block_depth(node: &ASTNode) -> usize {
                    match node {
                        ASTNode::ASTSquareBrackets(sb) => {
                            if sb.values.is_empty() {
                                1
                            } else {
                                1 + ast_block_depth(&sb.values[0])
                            }
                        }
                        ASTNode::ASTParenthesis(p) => {
                            if p.values.is_empty() {
                                1
                            } else {
                                1 + ast_block_depth(&p.values[0])
                            }
                        }
                        _ => 0,
                    }
                }
                let depth = ctx.ast_node.args.first().map(ast_block_depth).unwrap_or(0);
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::np_block_with_depth(&val, depth)
            }
            ("np", "reshape") => {
                // P4c: native StaticArray dispatch via array_ops::reshape fast-path.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. } | Value::DynamicNDArray(_)) {
                    return Some(crate::helpers::array_ops::reshape(
                        &mut self.builder,
                        &val_orig,
                        &visited_args_orig[1..],
                    ));
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_reshape(
                    &mut self.builder,
                    &val,
                    &visited_args[1..],
                )
            }
            ("np", "repeat") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_repeat(
                    &mut self.builder,
                    &val,
                    &visited_args[1..],
                    _visited_kwargs,
                )
            }

            // ── DynamicNDArray class methods ────────────────────────────
            ("DynamicNDArray", "zeros") | ("zinnia", "zeros") => {
                crate::ops::dyn_ndarray::metadata::dyn_zeros(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("DynamicNDArray", "ones") | ("zinnia", "ones") => {
                crate::ops::dyn_ndarray::metadata::dyn_ones(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("DynamicNDArray", "eye") | ("zinnia", "eye") => {
                crate::ops::dyn_ndarray::constructors::dyn_eye(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("DynamicNDArray", "concatenate") | ("zinnia", "concatenate") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            ("DynamicNDArray", "stack") | ("zinnia", "stack") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_stack(
                    &mut self.builder,
                    visited_args,
                    _visited_kwargs,
                )
            }
            _ => return None,
        };
        Some(v)
    }

    /// np.* fallback to the `np_like` registry. Anything under the `np` /
    /// `zinnia` / `math` namespace that wasn't matched by an explicit arm
    /// in `try_numpy_func` (or one of the explicit list / ndarray method
    /// arms in between) lands here. The registry plus centralised
    /// unary/binary vectorisation handles ops like np.sqrt, np.exp,
    /// np.sin, np.abs, np.minimum, np.equal, etc.
    pub(crate) fn try_np_fallback(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let target = ctx.target?;
        if !matches!(target, "np" | "zinnia" | "math") {
            return None;
        }
        let name = ctx.member;
        let member = ctx.member;
        let visited_args = ctx.args;
        let visited_args_orig = ctx.args_orig;
        let _visited_kwargs = ctx.kwargs;

        fn dispatch_scalar(
            builder: &mut crate::builder::IRBuilder,
            name: &str,
            kwargs: HashMap<String, Value>,
        ) -> Option<Value> {
            let op_args = crate::ops::OpArgs::new(kwargs);
            crate::ops::registry::build_op(
                name,
                Some(crate::ops::registry::OpNamespace::Np),
                builder,
                &op_args,
            )
        }
        fn vectorize_unary_np(
            builder: &mut crate::builder::IRBuilder,
            name: &str,
            base_kwargs: &HashMap<String, Value>,
            x: &Value,
        ) -> Value {
            match x {
                Value::List(d) | Value::Tuple(d) => {
                    let vals: Vec<Value> = d
                        .values
                        .iter()
                        .map(|v| vectorize_unary_np(builder, name, base_kwargs, v))
                        .collect();
                    let types = vals.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData {
                        elements_type: types,
                        values: vals,

                        value_id: ValueId::next(),
                    })
                }
                Value::DynamicNDArray(d) => {
                    let out_dtype = d.dtype;
                    let name_owned = name.to_string();
                    let kw_clone = base_kwargs.clone();
                    crate::ops::dyn_ndarray::binary::dyn_unary_op(
                        builder,
                        d,
                        out_dtype,
                        move |b, v| {
                            let mut kw = kw_clone.clone();
                            kw.insert("x".to_string(), v.clone());
                            dispatch_scalar(b, &name_owned, kw)
                                .unwrap_or_else(|| panic!("np.{} is not implemented", name_owned))
                        },
                    )
                }
                Value::Complex { real, imag } => {
                    // Complex transcendentals via Euler / standard identities,
                    // built on the existing real Float trig+exp+sqrt gates.
                    let re = Value::Float(real.clone());
                    let im = Value::Float(imag.clone());
                    match name {
                        "exp" => {
                            // exp(a + bi) = e^a * (cos(b) + i*sin(b))
                            let exp_a = builder.ir_exp_f(&re);
                            let cos_b = builder.ir_cos_f(&im);
                            let sin_b = builder.ir_sin_f(&im);
                            let r = builder.ir_mul_f(&exp_a, &cos_b);
                            let i = builder.ir_mul_f(&exp_a, &sin_b);
                            let r_sv = match r {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            let i_sv = match i {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            Value::Complex {
                                real: r_sv,
                                imag: i_sv,
                            }
                        }
                        "sin" => {
                            // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
                            let sa = builder.ir_sin_f(&re);
                            let ca = builder.ir_cos_f(&re);
                            let cb = builder.ir_cosh_f(&im);
                            let sb = builder.ir_sinh_f(&im);
                            let r = builder.ir_mul_f(&sa, &cb);
                            let i = builder.ir_mul_f(&ca, &sb);
                            let r_sv = match r {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            let i_sv = match i {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            Value::Complex {
                                real: r_sv,
                                imag: i_sv,
                            }
                        }
                        "cos" => {
                            // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
                            let ca = builder.ir_cos_f(&re);
                            let sa = builder.ir_sin_f(&re);
                            let cb = builder.ir_cosh_f(&im);
                            let sb = builder.ir_sinh_f(&im);
                            let r = builder.ir_mul_f(&ca, &cb);
                            let prod_sa_sb = builder.ir_mul_f(&sa, &sb);
                            let zero = builder.ir_constant_float(0.0);
                            let i_neg = builder.ir_sub_f(&zero, &prod_sa_sb);
                            let r_sv = match r {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            let i_sv = match i_neg {
                                Value::Float(s) => s,
                                _ => unreachable!(),
                            };
                            Value::Complex {
                                real: r_sv,
                                imag: i_sv,
                            }
                        }
                        "sqrt" => {
                            // sqrt(z) where z = a + bi:
                            // |z| = sqrt(a² + b²); when im == 0 and re ≥ 0, fall back
                            // to real sqrt with imag=0.
                            // Polar form: sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
                            // Implemented via the formula:
                            //   r = sqrt((|z| + a) / 2)
                            //   s = sign(b) * sqrt((|z| - a) / 2)
                            let aa = builder.ir_mul_f(&re, &re);
                            let bb = builder.ir_mul_f(&im, &im);
                            let mod_sq = builder.ir_add_f(&aa, &bb);
                            let modulus = builder.ir_sqrt_f(&mod_sq);
                            let two = builder.ir_constant_float(2.0);
                            let mod_plus_a = builder.ir_add_f(&modulus, &re);
                            let mod_minus_a = builder.ir_sub_f(&modulus, &re);
                            let half_plus = builder.ir_div_f(&mod_plus_a, &two);
                            let half_minus = builder.ir_div_f(&mod_minus_a, &two);
                            let r = builder.ir_sqrt_f(&half_plus);
                            let s_abs = builder.ir_sqrt_f(&half_minus);
                            // Sign of imag: if b >= 0 then +s_abs else -s_abs.
                            let zero = builder.ir_constant_float(0.0);
                            let pos = builder.ir_greater_than_or_equal_f(&im, &zero);
                            let neg_s = builder.ir_sub_f(&zero, &s_abs);
                            let s = builder.ir_select_f(&pos, &s_abs, &neg_s);
                            let r_sv = match r {
                                Value::Float(sv) => sv,
                                _ => unreachable!(),
                            };
                            let s_sv = match s {
                                Value::Float(sv) => sv,
                                _ => unreachable!(),
                            };
                            Value::Complex {
                                real: r_sv,
                                imag: s_sv,
                            }
                        }
                        "log" => {
                            // log(a + bi) = log(|z|) + i*arg(z) where arg = atan2(b, a)
                            let aa = builder.ir_mul_f(&re, &re);
                            let bb = builder.ir_mul_f(&im, &im);
                            let mod_sq = builder.ir_add_f(&aa, &bb);
                            let modulus = builder.ir_sqrt_f(&mod_sq);
                            let real = builder.ir_log_f(&modulus);
                            let imag = builder.ir_arctan2_f(&im, &re);
                            let r_sv = match real {
                                Value::Float(sv) => sv,
                                _ => unreachable!(),
                            };
                            let i_sv = match imag {
                                Value::Float(sv) => sv,
                                _ => unreachable!(),
                            };
                            Value::Complex {
                                real: r_sv,
                                imag: i_sv,
                            }
                        }
                        _ => {
                            panic!("np.{} on Complex is not yet implemented (compiler.complex-transcendentals scope)", name)
                        }
                    }
                }
                _ => {
                    let mut kw = base_kwargs.clone();
                    kw.insert("x".to_string(), x.clone());
                    dispatch_scalar(builder, name, kw)
                        .unwrap_or_else(|| panic!("np.{} is not implemented", name))
                }
            }
        }

        // Mapping for binary ops to the short names accepted by
        // `apply_binary_op`. Used for centralized binary
        // vectorization below.
        let binary_apply_name: Option<&str> = match member {
            "add" => Some("add"),
            "subtract" => Some("sub"),
            "multiply" => Some("mul"),
            "divide" => Some("div"),
            "floor_divide" => Some("floor_div"),
            "mod" | "fmod" => Some("mod"),
            "power" | "pow" => Some("pow"),
            "equal" => Some("eq"),
            "not_equal" => Some("ne"),
            "less" => Some("lt"),
            "less_equal" => Some("lte"),
            "greater" => Some("gt"),
            "greater_equal" => Some("gte"),
            "logical_and" => Some("and"),
            "logical_or" => Some("or"),
            _ => None,
        };

        // P5a: np.abs/absolute/fabs on a Complex StaticArray → fresh
        // Float StaticArray. Other np.* unary ops on Complex
        // StaticArray fall through to legacy path via deep_to_value_list
        // (for now; native vectorization is a follow-up).
        let result = if visited_args.len() == 1
            && matches!(
                visited_args_orig.first(),
                Some(Value::StaticArray {
                    dtype: crate::types::NumberType::Complex,
                    ..
                })
            )
            && matches!(member, "abs" | "absolute" | "fabs")
        {
            let arr = visited_args_orig[0].clone();
            Some(
                crate::helpers::static_array::complex::np_abs_complex_static_array(
                    &mut self.builder,
                    &arr,
                ),
            )
        } else if visited_args.len() == 1
            && matches!(
                visited_args[0],
                Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_) | Value::Complex { .. }
            )
        {
            // Centralized unary vectorization.
            let out = vectorize_unary_np(
                &mut self.builder,
                member,
                _visited_kwargs,
                &visited_args[0],
            );
            // Keep ndarray-ness: a StaticArray input must yield a
            // StaticArray result — a numeric List result would pick up
            // Python sequence semantics downstream (`sin(a) + sin(b)`
            // would CONCATENATE).
            let out = if matches!(visited_args_orig.first(), Some(Value::StaticArray { .. })) {
                crate::helpers::static_array::base::to_static_array(&mut self.builder, &out)
                    .unwrap_or(out)
            } else {
                out
            };
            Some(out)
        } else if visited_args.len() == 2
            && (matches!(
                visited_args_orig[0],
                Value::StaticArray { .. } | Value::Complex { .. }
            ) || matches!(
                visited_args_orig[1],
                Value::StaticArray { .. } | Value::Complex { .. }
            ))
            && binary_apply_name.is_some()
        {
            Some(crate::helpers::value_ops::apply_binary_op(
                &mut self.builder,
                binary_apply_name.unwrap(),
                &visited_args_orig[0],
                &visited_args_orig[1],
            ))
        } else if visited_args.len() == 2
            && (matches!(
                visited_args[0],
                Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_)
            ) || matches!(
                visited_args[1],
                Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_)
            ))
            && binary_apply_name.is_some()
        {
            // Centralized binary vectorization for composite operands.
            // Falls through to `apply_binary_op` which already does
            // shape-broadcasting and dtype promotion.
            Some(crate::helpers::value_ops::apply_binary_op(
                &mut self.builder,
                binary_apply_name.unwrap(),
                &visited_args[0],
                &visited_args[1],
            ))
        } else {
            let mut all_kwargs = _visited_kwargs.clone();
            if visited_args.len() == 1 {
                all_kwargs
                    .entry("x".to_string())
                    .or_insert_with(|| visited_args[0].clone());
            } else {
                for (i, v) in visited_args.iter().enumerate() {
                    let key = format!("x{}", i + 1);
                    all_kwargs.entry(key).or_insert_with(|| v.clone());
                }
            }
            dispatch_scalar(&mut self.builder, member, all_kwargs)
        };
        match result {
            Some(v) => Some(v),
            None => panic!("np.{} is not implemented", name),
        }
    }
}
