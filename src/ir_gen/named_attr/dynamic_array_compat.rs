//! NumPy rank-one operations whose result or operand length is witness-dependent.
//!
//! Capacity is never treated as the logical iteration length. Gathers use a
//! one-hot selection network, so inactive padding cannot issue invalid reads.
use crate::builder::IRBuilder;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, Envelope, NumberType, ScalarValue, Value, ValueId,
};
use std::collections::HashMap;

fn vector(b: &mut IRBuilder, value: &Value, op: &str) -> DynamicNDArrayData {
    let data = match value {
        Value::DynamicNDArray(d) => d.clone(),
        Value::StaticArray { shape, dtype, .. } => {
            // The legacy promotion helper flattens composites without a
            // builder and cannot read segment-backed StaticArray payloads.
            // Read the actual view cells (including offset/stride) here.
            let payload = crate::helpers::static_array::elementwise::payload_cells(b, value);
            let length = shape.iter().product::<usize>();
            let strides = crate::helpers::shape_arith::row_major_strides(shape);
            DynamicNDArrayData {
                envelope: Envelope::from_static_shape(&mut b.dim_table, shape),
                dtype: *dtype,
                segment_id: crate::helpers::segment::alloc_and_write_values(b, &payload, *dtype),
                meta: DynArrayMeta {
                    logical_shape: shape.clone(),
                    logical_offset: 0,
                    logical_strides: strides.clone(),
                    runtime_length: ScalarValue::new(Some(length as i64), None),
                    runtime_rank: ScalarValue::new(Some(shape.len() as i64), None),
                    runtime_shape: shape
                        .iter()
                        .map(|&n| ScalarValue::new(Some(n as i64), None))
                        .collect(),
                    runtime_strides: strides
                        .iter()
                        .map(|&n| ScalarValue::new(Some(n as i64), None))
                        .collect(),
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
                value_id: ValueId::next(),
            }
        }
        _ => match crate::helpers::promote::promote_static_to_dynamic(b, value) {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        },
    };
    assert_eq!(
        data.envelope.rank(),
        1,
        "np.{op}: this bounded-array operation requires rank one"
    );
    crate::ops::dyn_ndarray::force_contiguous(b, &data)
}

fn cells(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Vec<Value> {
    crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype)
}

fn zero(b: &mut IRBuilder, dtype: NumberType) -> Value {
    crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype)
}

fn arg<'a>(
    args: &'a [Value],
    kwargs: &'a HashMap<String, Value>,
    index: usize,
    name: &str,
) -> Option<&'a Value> {
    assert!(
        !(args.get(index).is_some() && kwargs.contains_key(name)),
        "multiple values for {name}"
    );
    args.get(index).or_else(|| kwargs.get(name))
}

fn rank_one_axis(axis: Option<&Value>, op: &str) {
    if let Some(axis) = axis.filter(|v| !matches!(v, Value::None)) {
        assert!(
            matches!(axis.int_val(), Some(0 | -1)),
            "np.{op}: bounded rank-one axis must be 0, -1, or None"
        );
    }
}

fn dot(b: &mut IRBuilder, lhs: &Value, rhs: &Value, op: &str) -> Value {
    let left = vector(b, lhs, op);
    let right = vector(b, rhs, op);
    let left_len = Value::Integer(left.meta.runtime_length.clone());
    let right_len = Value::Integer(right.meta.runtime_length.clone());
    let same_length = b.ir_equal_i(&left_len, &right_len);
    b.ir_assert(&same_length);
    let left_cells = cells(b, &left);
    let right_cells = cells(b, &right);
    let dtype = if left.dtype == NumberType::Complex || right.dtype == NumberType::Complex {
        NumberType::Complex
    } else if left.dtype == NumberType::Float || right.dtype == NumberType::Float {
        NumberType::Float
    } else {
        NumberType::Integer
    };
    let identity = zero(b, dtype);
    let mut result = identity.clone();
    // Equal logical lengths need not imply equal public capacities.
    for (i, (a, c)) in left_cells.iter().zip(right_cells.iter()).enumerate() {
        let k = b.ir_constant_int(i as i64);
        let active = b.ir_less_than_i(&k, &left_len);
        let product = crate::helpers::value_ops::apply_binary_op(b, "mul", a, c);
        let contribution = crate::helpers::value_ops::select_value(b, &active, &product, &identity);
        result = crate::helpers::value_ops::apply_binary_op(b, "add", &result, &contribution);
    }
    result
}

fn take(b: &mut IRBuilder, source: &Value, indices: &Value) -> Value {
    let source = vector(b, source, "take");
    let indices = vector(b, indices, "take");
    assert_eq!(
        indices.dtype,
        NumberType::Integer,
        "np.take: integer indices required"
    );
    let source_cells = cells(b, &source);
    let index_cells = cells(b, &indices);
    let source_len = Value::Integer(source.meta.runtime_length.clone());
    let index_len = Value::Integer(indices.meta.runtime_length.clone());
    let identity = zero(b, source.dtype);
    let z = b.ir_constant_int(0);
    let mut output = Vec::with_capacity(index_cells.len());
    for (i, index) in index_cells.iter().enumerate() {
        let k = b.ir_constant_int(i as i64);
        let active = b.ir_less_than_i(&k, &index_len);
        let negative = b.ir_less_than_i(index, &z);
        let wrapped = b.ir_add_i(index, &source_len);
        let normalized = b.ir_select_i(&negative, &wrapped, index);
        let below_zero = b.ir_less_than_i(&normalized, &z);
        let nonnegative = b.ir_logical_not(&below_zero);
        let below_length = b.ir_less_than_i(&normalized, &source_len);
        let in_bounds = b.ir_logical_and(&nonnegative, &below_length);
        let inactive = b.ir_logical_not(&active);
        let valid = b.ir_logical_or(&inactive, &in_bounds);
        b.ir_assert(&valid);
        let mut selected = identity.clone();
        // Reading each physical source cell is always safe, even when its
        // logical array is empty. No witness-dependent memory read is issued.
        for (j, cell) in source_cells.iter().enumerate() {
            let address = b.ir_constant_int(j as i64);
            let matches = b.ir_equal_i(&normalized, &address);
            let selected_here = b.ir_logical_and(&active, &matches);
            selected = crate::helpers::value_ops::select_value(b, &selected_here, cell, &selected);
        }
        output.push(selected);
    }
    let mut result = indices;
    result.dtype = source.dtype;
    result.segment_id = crate::helpers::segment::alloc_and_write_values(b, &output, result.dtype);
    result.value_id = ValueId::next();
    Value::DynamicNDArray(result)
}

fn compress(b: &mut IRBuilder, condition: &Value, source: &Value) -> Value {
    let source = vector(b, source, "compress");
    let condition = vector(b, condition, "compress");
    let conditions = cells(b, &condition);
    let condition_len = Value::Integer(condition.meta.runtime_length.clone());
    let source_len = Value::Integer(source.meta.runtime_length.clone());
    let z = b.ir_constant_int(0);
    let mut mask = vec![z.clone(); source.max_length()];
    for (i, cell) in conditions.iter().enumerate() {
        let k = b.ir_constant_int(i as i64);
        let condition_active = b.ir_less_than_i(&k, &condition_len);
        let truth = crate::helpers::value_ops::to_scalar_bool(b, cell);
        let keep = b.ir_logical_and(&condition_active, &truth);
        // NumPy permits a shorter condition (truncating the source), and
        // longer conditions provided they select no out-of-bounds position.
        let source_active = b.ir_less_than_i(&k, &source_len);
        let not_kept = b.ir_logical_not(&keep);
        let valid = b.ir_logical_or(&not_kept, &source_active);
        b.ir_assert(&valid);
        if i < mask.len() {
            mask[i] = keep;
        }
    }
    let mask = crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        mask,
        vec![source.max_length()],
        NumberType::Integer,
    );
    crate::ops::dyn_ndarray::memory_ops::dyn_filter(b, &source, &[mask])
}

pub(super) fn dispatch(
    b: &mut IRBuilder,
    member: &str,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Option<Value> {
    match member {
        "dot" | "inner" => {
            let lhs = arg(args, kwargs, 0, "a")?;
            let rhs = arg(args, kwargs, 1, "b")?;
            if !matches!(lhs, Value::DynamicNDArray(_)) && !matches!(rhs, Value::DynamicNDArray(_))
            {
                return None;
            }
            assert!(
                args.len() <= if member == "dot" { 3 } else { 2 },
                "np.{member}: too many arguments"
            );
            assert!(
                kwargs
                    .keys()
                    .all(|k| ["a", "b", "out"].contains(&k.as_str())),
                "np.{member}: unsupported keyword"
            );
            assert!(
                member != "inner" || !kwargs.contains_key("out"),
                "np.inner does not support out"
            );
            let output = dot(b, lhs, rhs, member);
            if let Some(out) = arg(args, kwargs, 2, "out").filter(|v| !matches!(v, Value::None)) {
                return Some(super::array_compat::write_output(b, &output, out, None));
            }
            Some(output)
        }
        "compress" => {
            assert!(args.len() <= 4, "np.compress: too many arguments");
            assert!(
                kwargs
                    .keys()
                    .all(|k| ["condition", "a", "axis", "out"].contains(&k.as_str())),
                "np.compress: unsupported keyword"
            );
            let condition = arg(args, kwargs, 0, "condition")?;
            let source = arg(args, kwargs, 1, "a")?;
            rank_one_axis(arg(args, kwargs, 2, "axis"), member);
            assert!(
                arg(args, kwargs, 3, "out").is_none_or(|v| matches!(v, Value::None)),
                "np.compress: bounded output does not support out"
            );
            Some(compress(b, condition, source))
        }
        "take" => {
            let source = arg(args, kwargs, 0, "a")?;
            let indices = arg(args, kwargs, 1, "indices")?;
            if !matches!(indices, Value::DynamicNDArray(_)) {
                return None;
            }
            assert!(args.len() <= 5, "np.take: too many arguments");
            assert!(
                kwargs
                    .keys()
                    .all(|k| ["a", "indices", "axis", "out", "mode"].contains(&k.as_str())),
                "np.take: unsupported keyword"
            );
            rank_one_axis(arg(args, kwargs, 2, "axis"), member);
            assert!(
                arg(args, kwargs, 3, "out").is_none_or(|v| matches!(v, Value::None)),
                "np.take: bounded output does not support out"
            );
            if let Some(mode) = arg(args, kwargs, 4, "mode") {
                assert_eq!(
                    mode.string_val(),
                    Some("raise"),
                    "np.take: bounded indices currently require mode='raise'"
                );
            }
            Some(take(b, source, indices))
        }
        _ => None,
    }
}
