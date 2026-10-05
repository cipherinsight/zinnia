//! Static-shape NumPy iteration, scans, segmented reductions, and neighborhood views.
//!
//! The frontend maps nested NumPy/ufunc names to the members accepted by
//! `dispatch`. Inputs remain StaticArray values throughout this module; no
//! array is lowered to a Python list before arithmetic or view construction.
use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, row_major_strides};
use crate::helpers::static_array::base::{
    build_static_array_from_flat, build_static_array_from_flat_complex, to_static_array,
};
use crate::helpers::static_array::elementwise::payload_cells;
use crate::types::{CompositeData, NumberType, Value, ValueId, ZinniaType};

fn known_int(value: &Value, op: &str) -> i64 {
    value
        .int_val()
        .unwrap_or_else(|| panic!("{op}: requires a compile-time integer"))
}

fn integers(b: &mut IRBuilder, value: &Value, op: &str) -> Vec<i64> {
    match value {
        Value::List(d) | Value::Tuple(d) => d.values.iter().map(|v| known_int(v, op)).collect(),
        Value::StaticArray { shape, .. } => {
            assert!(
                shape.len() <= 1,
                "{op}: expected a scalar or one-dimensional integer sequence"
            );
            payload_cells(b, value)
                .iter()
                .map(|v| known_int(v, op))
                .collect()
        }
        _ => vec![known_int(value, op)],
    }
}

fn array(b: &mut IRBuilder, value: &Value, op: &str) -> Value {
    if matches!(value, Value::StaticArray { .. }) {
        return value.clone();
    }
    assert!(
        !matches!(value, Value::DynamicNDArray(_)),
        "{op}: runtime array dimensions are not supported"
    );
    to_static_array(b, value)
        .unwrap_or_else(|| panic!("{op}: requires a rectangular numeric array"))
}

fn metadata(value: &Value) -> (Vec<usize>, NumberType) {
    let Value::StaticArray { shape, dtype, .. } = value else {
        unreachable!()
    };
    (shape.clone(), *dtype)
}

fn size(shape: &[usize], op: &str) -> usize {
    shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .unwrap_or_else(|| panic!("{op}: array shape is too large"))
}

fn axis(value: i64, rank: usize, op: &str) -> usize {
    let result = if value < 0 {
        value + rank as i64
    } else {
        value
    };
    assert!(
        result >= 0 && result < rank as i64,
        "{op}: axis {value} is out of bounds for rank {rank}"
    );
    result as usize
}

fn arguments(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    names: &[&str],
    op: &str,
) -> HashMap<String, Value> {
    assert!(
        args.len() <= names.len(),
        "{op}: too many positional arguments"
    );
    let mut result = kwargs.clone();
    for key in result.keys() {
        assert!(
            names.contains(&key.as_str()),
            "{op}: unsupported keyword {key}"
        );
    }
    for (name, value) in names.iter().zip(args) {
        assert!(
            !result.contains_key(*name),
            "{op}: multiple values for {name}"
        );
        result.insert((*name).to_string(), value.clone());
    }
    result
}

fn required<'a>(values: &'a HashMap<String, Value>, name: &str, op: &str) -> &'a Value {
    values
        .get(name)
        .unwrap_or_else(|| panic!("{op}: missing required argument {name}"))
}

fn build(b: &mut IRBuilder, cells: Vec<Value>, shape: Vec<usize>, dtype: NumberType) -> Value {
    if dtype == NumberType::Complex {
        let mut real = Vec::with_capacity(cells.len());
        let mut imag = Vec::with_capacity(cells.len());
        for cell in cells {
            let Value::Complex { real: r, imag: i } = cell else {
                panic!("complex array contains a non-complex cell")
            };
            real.push(Value::Float(r));
            imag.push(Value::Float(i));
        }
        build_static_array_from_flat_complex(b, real, imag, shape)
    } else {
        build_static_array_from_flat(b, cells, shape, dtype)
    }
}

fn finish(b: &mut IRBuilder, result: Value, params: &HashMap<String, Value>) -> Value {
    if let Some(out) = params.get("out").filter(|v| !matches!(v, Value::None)) {
        super::array_compat::write_output(b, &result, out, None)
    } else {
        result
    }
}

fn same_dtype(params: &HashMap<String, Value>, input_dtype: NumberType, op: &str) {
    let requested = match params.get("dtype") {
        None | Some(Value::None) => return,
        Some(Value::Class(ZinniaType::Integer)) => NumberType::Integer,
        Some(Value::Class(ZinniaType::Float)) => NumberType::Float,
        Some(Value::Class(ZinniaType::Complex)) => NumberType::Complex,
        Some(Value::String(value)) => match value.val.as_str() {
            "int" | "int64" | "i8" => NumberType::Integer,
            "float" | "float64" | "f8" => NumberType::Float,
            "complex" | "complex128" | "c16" => NumberType::Complex,
            other => panic!("{op}: unsupported accumulator dtype {other}"),
        },
        _ => panic!("{op}: unsupported accumulator dtype"),
    };
    assert_eq!(
        requested, input_dtype,
        "{op}: converting the accumulator dtype is not supported"
    );
}

fn ndindex(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    assert!(
        kwargs.is_empty(),
        "np.ndindex: keyword arguments are not supported"
    );
    let dims = if args.len() == 1 {
        integers(b, &args[0], "np.ndindex")
    } else {
        args.iter().map(|v| known_int(v, "np.ndindex")).collect()
    };
    assert!(
        dims.iter().all(|d| *d >= 0),
        "np.ndindex: negative dimensions are not allowed"
    );
    let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    let strides = row_major_strides(&shape);
    let values: Vec<Value> = (0..size(&shape, "np.ndindex"))
        .map(|i| {
            let values: Vec<Value> = decode_coords(i, &shape, &strides)
                .into_iter()
                .map(|c| b.ir_constant_int(c as i64))
                .collect();
            Value::Tuple(CompositeData {
                elements_type: values.iter().map(Value::zinnia_type).collect(),
                values,
                value_id: ValueId::next(),
            })
        })
        .collect();
    Value::List(CompositeData {
        elements_type: values.iter().map(Value::zinnia_type).collect(),
        values,
        value_id: ValueId::next(),
    })
}

fn combine(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    match op {
        "add" => crate::helpers::value_ops::apply_binary_op(b, "add", lhs, rhs),
        "minimum" => crate::ops::np_like::vectorize_minmax(
            b,
            lhs,
            rhs,
            IRBuilder::ir_less_than_i,
            IRBuilder::ir_less_than_f,
            false,
        ),
        "maximum" => crate::ops::np_like::vectorize_minmax(
            b,
            lhs,
            rhs,
            IRBuilder::ir_greater_than_i,
            IRBuilder::ir_greater_than_f,
            true,
        ),
        _ => unreachable!(),
    }
}

fn accumulate(
    b: &mut IRBuilder,
    op: &str,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let params = arguments(args, kwargs, &["array", "axis", "dtype", "out"], op);
    let input = array(b, required(&params, "array", op), op);
    let (shape, dtype) = metadata(&input);
    same_dtype(&params, dtype, op);
    assert!(
        dtype != NumberType::Complex,
        "{op}.accumulate: complex ordering is not supported"
    );
    let ax = axis(
        params.get("axis").map(|v| known_int(v, op)).unwrap_or(0),
        shape.len(),
        op,
    );
    let strides = row_major_strides(&shape);
    let input_cells = payload_cells(b, &input);
    let mut cells: Vec<Value> = Vec::with_capacity(input_cells.len());
    for (i, value) in input_cells.iter().enumerate() {
        let coordinate = (i / strides[ax]) % shape[ax];
        let next = if coordinate == 0 {
            value.clone()
        } else {
            combine(b, op, &cells[i - strides[ax]], value)
        };
        cells.push(next);
    }
    let result = build(b, cells, shape, dtype);
    finish(b, result, &params)
}

fn reduceat(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let op = "np.add.reduceat";
    let params = arguments(
        args,
        kwargs,
        &["array", "indices", "axis", "dtype", "out"],
        op,
    );
    let input = array(b, required(&params, "array", op), op);
    let (shape, dtype) = metadata(&input);
    same_dtype(&params, dtype, op);
    let ax = axis(
        params.get("axis").map(|v| known_int(v, op)).unwrap_or(0),
        shape.len(),
        op,
    );
    let indices_arg = required(&params, "indices", op);
    assert!(
        matches!(
            indices_arg,
            Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. }
        ),
        "{op}: indices must be a one-dimensional integer sequence"
    );
    let indices = integers(b, indices_arg, op);
    for start in &indices {
        assert!(
            *start >= 0 && (*start as usize) < shape[ax],
            "{op}: index is out of bounds"
        );
    }
    let input_cells = payload_cells(b, &input);
    let mut output_shape = shape.clone();
    output_shape[ax] = indices.len();
    let input_strides = row_major_strides(&shape);
    let output_strides = row_major_strides(&output_shape);
    let mut cells = Vec::with_capacity(size(&output_shape, op));
    for i in 0..size(&output_shape, op) {
        let mut coords = decode_coords(i, &output_shape, &output_strides);
        let segment = coords[ax];
        let start = indices[segment] as usize;
        let next = indices
            .get(segment + 1)
            .map(|v| *v as usize)
            .unwrap_or(shape[ax]);
        // NumPy emits a single input element when successive indices do
        // not increase, rather than an empty reduction.
        let stop = if next <= start { start + 1 } else { next };
        coords[ax] = start;
        let base: usize = coords.iter().zip(&input_strides).map(|(c, s)| c * s).sum();
        let mut result = input_cells[base].clone();
        for delta in 1..stop - start {
            result = combine(
                b,
                "add",
                &result,
                &input_cells[base + delta * input_strides[ax]],
            );
        }
        cells.push(result);
    }
    let result = build(b, cells, output_shape, dtype);
    finish(b, result, &params)
}

fn sliding_window(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let op = "np.lib.stride_tricks.sliding_window_view";
    assert!(
        args.len() <= 3,
        "{op}: subok and writeable are keyword-only"
    );
    let params = arguments(
        args,
        kwargs,
        &["x", "window_shape", "axis", "subok", "writeable"],
        op,
    );
    for key in ["subok", "writeable"] {
        if let Some(value) = params.get(key) {
            let flag = value
                .bool_val()
                .or_else(|| value.int_val().map(|n| n != 0))
                .unwrap_or_else(|| panic!("{op}: {key} must be a static Boolean"));
            assert!(!flag, "{op}: {key}=True is not supported");
        }
    }
    let input = array(b, required(&params, "x", op), op);
    let Value::StaticArray {
        dtype,
        shape,
        segment_id,
        strides,
        offset,
        imag_segment_id,
        ..
    } = input
    else {
        unreachable!()
    };
    let windows = integers(b, required(&params, "window_shape", op), op);
    assert!(
        windows.iter().all(|w| *w >= 0),
        "{op}: window dimensions cannot be negative"
    );
    let axes: Vec<usize> = match params.get("axis") {
        None | Some(Value::None) => (0..shape.len()).collect(),
        Some(v) => integers(b, v, op)
            .into_iter()
            .map(|a| axis(a, shape.len(), op))
            .collect(),
    };
    assert_eq!(
        windows.len(),
        axes.len(),
        "{op}: window_shape and axis lengths differ"
    );
    let mut output_shape = shape;
    let mut output_strides = strides.clone();
    // Repeated axes are legal: each successive window trims the current
    // extent, and each window dimension uses the same source-axis stride.
    for (window, ax) in windows.iter().zip(&axes) {
        let window = *window as usize;
        assert!(
            window <= output_shape[*ax],
            "{op}: window shape exceeds the input extent"
        );
        output_shape[*ax] = output_shape[*ax]
            .checked_add(1)
            .and_then(|d| d.checked_sub(window))
            .expect("window extent overflow");
        output_strides.push(strides[*ax]);
    }
    output_shape.extend(windows.into_iter().map(|v| v as usize));
    size(&output_shape, op);
    // A metadata-only view preserves offset, arbitrary positive strides,
    // and source aliases. Downstream native reads honor these strides.
    Value::StaticArray {
        dtype,
        shape: output_shape,
        segment_id,
        strides: output_strides,
        offset,
        imag_segment_id,
        value_id: ValueId::next(),
    }
}

fn roll(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let op = "np.roll";
    let params = arguments(args, kwargs, &["a", "shift", "axis"], op);
    let input = array(b, required(&params, "a", op), op);
    let (shape, dtype) = metadata(&input);
    let input_cells = payload_cells(b, &input);
    let shifts = integers(b, required(&params, "shift", op), op);
    let mut cells = Vec::with_capacity(input_cells.len());
    if matches!(params.get("axis"), None | Some(Value::None)) {
        let total = input_cells.len();
        if total != 0 {
            let shift = shifts
                .iter()
                .fold(0i128, |a, s| a + *s as i128)
                .rem_euclid(total as i128) as usize;
            for i in 0..total {
                cells.push(input_cells[(i + total - shift) % total].clone());
            }
        }
    } else {
        let axes = integers(b, params.get("axis").unwrap(), op);
        assert!(
            shifts.len() == axes.len() || shifts.len() == 1 || axes.len() == 1,
            "{op}: shift and axis cannot be broadcast together"
        );
        let count = if shifts.is_empty() || axes.is_empty() {
            0
        } else {
            shifts.len().max(axes.len())
        };
        let mut offsets = vec![0i128; shape.len()];
        for i in 0..count {
            let ax = axis(axes[if axes.len() == 1 { 0 } else { i }], shape.len(), op);
            offsets[ax] += shifts[if shifts.len() == 1 { 0 } else { i }] as i128;
        }
        let strides = row_major_strides(&shape);
        for i in 0..input_cells.len() {
            let mut coords = decode_coords(i, &shape, &strides);
            for (ax, coord) in coords.iter_mut().enumerate() {
                *coord = (*coord as i128 - offsets[ax]).rem_euclid(shape[ax] as i128) as usize;
            }
            let source: usize = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            cells.push(input_cells[source].clone());
        }
    }
    build(b, cells, shape, dtype)
}

fn pad_pairs(b: &mut IRBuilder, value: &Value, rank: usize) -> Vec<(usize, usize)> {
    let value = crate::helpers::static_array::base::deep_to_value_list(b, value);
    let pair = |v: &Value| -> (usize, usize) {
        let vals: Vec<i64> = match v {
            Value::Tuple(d) | Value::List(d) => {
                d.values.iter().map(|v| known_int(v, "np.pad")).collect()
            }
            _ => vec![known_int(v, "np.pad")],
        };
        assert!(
            vals.len() == 1 || vals.len() == 2,
            "np.pad: pad_width entries must be scalar or pairs"
        );
        assert!(
            vals.iter().all(|n| *n >= 0),
            "np.pad: negative pad widths are not allowed"
        );
        (vals[0] as usize, vals[vals.len() - 1] as usize)
    };
    match &value {
        Value::Tuple(d) | Value::List(d)
            if d.values
                .iter()
                .any(|v| matches!(v, Value::Tuple(_) | Value::List(_))) =>
        {
            assert!(
                d.values.len() == 1 || d.values.len() == rank,
                "np.pad: pad_width does not match array rank"
            );
            (0..rank)
                .map(|i| pair(&d.values[if d.values.len() == 1 { 0 } else { i }]))
                .collect()
        }
        _ => vec![pair(&value); rank],
    }
}

fn pad_edge(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let op = "np.pad(mode='edge')";
    let params = arguments(args, kwargs, &["array", "pad_width", "mode"], op);
    let input = array(b, required(&params, "array", op), op);
    let (shape, dtype) = metadata(&input);
    let pads = pad_pairs(b, required(&params, "pad_width", op), shape.len());
    let mut output_shape = Vec::with_capacity(shape.len());
    for (dim, (before, after)) in shape.iter().zip(&pads) {
        assert!(
            *dim != 0 || (*before == 0 && *after == 0),
            "{op}: cannot extend an empty axis"
        );
        output_shape.push(
            dim.checked_add(*before)
                .and_then(|n| n.checked_add(*after))
                .expect("np.pad: extent overflow"),
        );
    }
    let input_cells = payload_cells(b, &input);
    let source_strides = row_major_strides(&shape);
    let output_strides = row_major_strides(&output_shape);
    let mut cells = Vec::with_capacity(size(&output_shape, op));
    for i in 0..size(&output_shape, op) {
        let coords = decode_coords(i, &output_shape, &output_strides);
        let source: usize = coords
            .iter()
            .zip(&pads)
            .zip(&shape)
            .zip(&source_strides)
            .map(|(((c, (before, _)), dim), stride)| {
                c.saturating_sub(*before).min(dim - 1) * stride
            })
            .sum();
        cells.push(input_cells[source].clone());
    }
    build(b, cells, output_shape, dtype)
}

pub(super) fn dispatch(
    b: &mut IRBuilder,
    member: &str,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Option<Value> {
    Some(match member {
        "ndindex" => ndindex(b, args, kwargs),
        "minimum_accumulate" => accumulate(b, "minimum", args, kwargs),
        "maximum_accumulate" => accumulate(b, "maximum", args, kwargs),
        "add_reduceat" => reduceat(b, args, kwargs),
        "sliding_window_view" => sliding_window(b, args, kwargs),
        "roll" => roll(b, args, kwargs),
        "pad"
            if args
                .get(2)
                .or_else(|| kwargs.get("mode"))
                .and_then(Value::string_val)
                == Some("edge") =>
        {
            pad_edge(b, args, kwargs)
        }
        _ => return None,
    })
}
