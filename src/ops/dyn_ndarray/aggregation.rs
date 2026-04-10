use crate::builder::IRBuilder;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value,
};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    scalar_i64_to_value, value_to_scalar_i64, DynAggKind,
};

pub fn dyn_aggregate(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    axis: Option<&Value>,
    agg: DynAggKind,
) -> Value {
    let axis_val = axis.and_then(|v| {
        if matches!(v, Value::None) {
            None
        } else {
            v.int_val()
        }
    });

    match axis_val {
        None => dyn_aggregate_all(b, data, agg),
        Some(ax) => dyn_aggregate_axis(b, data, ax, agg),
    }
}

/// Full reduction (axis=None): reduce all elements to a scalar.
pub fn dyn_aggregate_all(b: &mut IRBuilder, data: &DynamicNDArrayData, agg: DynAggKind) -> Value {
    let values = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let numel = dyn_num_elements(&data.meta.logical_shape);
    if numel == 0 {
        return dyn_agg_identity(b, agg, data.dtype);
    }

    let use_float = data.dtype == NumberType::Float
        && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    let mut acc = values[0].clone();
    let mut acc_idx = b.ir_constant_int(0);

    for i in 1..numel.min(values.len()) {
        let elem = values[i].clone();
        let idx_val = b.ir_constant_int(i as i64);
        let (new_acc, new_idx) =
            dyn_agg_step(b, &acc, &acc_idx, &elem, &idx_val, agg, use_float);
        acc = new_acc;
        acc_idx = new_idx;
    }

    // For argmax/argmin, return the index
    match agg {
        DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
        _ => acc,
    }
}

/// Axis reduction: reduce along a specific axis.
pub fn dyn_aggregate_axis(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    axis: i64,
    agg: DynAggKind,
) -> Value {
    let shape = &data.meta.logical_shape.clone();
    let ndim = shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax < ndim, "aggregate axis out of bounds");

    let values = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let strides = dyn_row_major_strides(shape);
    let use_float = data.dtype == NumberType::Float
        && !matches!(agg, DynAggKind::All | DynAggKind::Any);

    // Output shape: remove the reduced axis
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != ax)
        .map(|(_, &s)| s)
        .collect();
    let out_numel: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let out_strides = dyn_row_major_strides(&out_shape);
    let axis_dim = shape[ax];

    let mut out_elements = Vec::with_capacity(out_numel);

    for out_idx in 0..out_numel {
        // Decode output coordinates
        let out_coords = if out_shape.is_empty() {
            vec![]
        } else {
            dyn_decode_coords(out_idx, &out_shape, &out_strides)
        };

        // Build input coordinates: insert axis position
        let mut in_coords = out_coords.clone();
        in_coords.insert(ax, 0);

        // Initialize accumulator with first element along axis
        let first_src_idx = dyn_encode_coords(&in_coords, &strides);
        let first_elem = if first_src_idx < values.len() {
            values[first_src_idx].clone()
        } else {
            super::metadata::dyn_default_value(b, data.dtype)
        };

        let mut acc = first_elem;
        let mut acc_idx = b.ir_constant_int(0);

        // Iterate along reduction axis
        for k in 1..axis_dim {
            in_coords[ax] = k;
            let src_idx = dyn_encode_coords(&in_coords, &strides);
            let elem = if src_idx < values.len() {
                values[src_idx].clone()
            } else {
                super::metadata::dyn_default_value(b, data.dtype)
            };
            let k_val = b.ir_constant_int(k as i64);
            let (new_acc, new_idx) =
                dyn_agg_step(b, &acc, &acc_idx, &elem, &k_val, agg, use_float);
            acc = new_acc;
            acc_idx = new_idx;
        }

        let result = match agg {
            DynAggKind::Argmax | DynAggKind::Argmin => acc_idx,
            _ => acc,
        };
        out_elements.push(value_to_scalar_i64(&result));
    }

    // Determine output dtype
    let out_dtype = match agg {
        DynAggKind::All | DynAggKind::Any | DynAggKind::Argmax | DynAggKind::Argmin => {
            NumberType::Integer
        }
        _ => data.dtype,
    };

    if out_shape.is_empty() {
        // Scalar result
        return scalar_i64_to_value(&out_elements[0], out_dtype);
    }

    let out_strides_meta = dyn_row_major_strides(&out_shape);
    let _ = out_numel;
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, out_dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides_meta.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: out_strides_meta
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    Value::DynamicNDArray(result)
}

/// One step of the accumulator for a given aggregation kind.
pub fn dyn_agg_step(
    b: &mut IRBuilder,
    acc: &Value,
    acc_idx: &Value,
    elem: &Value,
    elem_idx: &Value,
    agg: DynAggKind,
    use_float: bool,
) -> (Value, Value) {
    match agg {
        DynAggKind::Sum => {
            let new_acc = if use_float {
                b.ir_add_f(acc, elem)
            } else {
                b.ir_add_i(acc, elem)
            };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Prod => {
            let new_acc = if use_float {
                b.ir_mul_f(acc, elem)
            } else {
                b.ir_mul_i(acc, elem)
            };
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Max => {
            let cond = if use_float {
                b.ir_greater_than_f(elem, acc)
            } else {
                b.ir_greater_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::Min => {
            let cond = if use_float {
                b.ir_less_than_f(elem, acc)
            } else {
                b.ir_less_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::All => {
            let bv = b.ir_bool_cast(elem);
            let new_acc = b.ir_logical_and(acc, &bv);
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Any => {
            let bv = b.ir_bool_cast(elem);
            let new_acc = b.ir_logical_or(acc, &bv);
            (new_acc, acc_idx.clone())
        }
        DynAggKind::Argmax => {
            let cond = if use_float {
                b.ir_greater_than_f(elem, acc)
            } else {
                b.ir_greater_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
        DynAggKind::Argmin => {
            let cond = if use_float {
                b.ir_less_than_f(elem, acc)
            } else {
                b.ir_less_than_i(elem, acc)
            };
            let new_acc = if use_float {
                b.ir_select_f(&cond, elem, acc)
            } else {
                b.ir_select_i(&cond, elem, acc)
            };
            let new_idx = b.ir_select_i(&cond, elem_idx, acc_idx);
            (new_acc, new_idx)
        }
    }
}

/// Identity value for aggregation init (used when array is empty).
pub fn dyn_agg_identity(b: &mut IRBuilder, agg: DynAggKind, dtype: NumberType) -> Value {
    match agg {
        DynAggKind::Sum => super::metadata::dyn_default_value(b, dtype),
        DynAggKind::Prod => match dtype {
            NumberType::Integer => b.ir_constant_int(1),
            NumberType::Float => b.ir_constant_float(1.0),
        },
        DynAggKind::All => b.ir_constant_bool(true),
        DynAggKind::Any => b.ir_constant_bool(false),
        DynAggKind::Max | DynAggKind::Min => super::metadata::dyn_default_value(b, dtype),
        DynAggKind::Argmax | DynAggKind::Argmin => b.ir_constant_int(0),
    }
}
