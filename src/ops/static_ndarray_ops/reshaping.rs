//! Shape / rank manipulation, broadcasting, joining (concat/stack family),
//! and splitting (split/array_split/block family). All of these operate
//! on the composite-Value layer rather than at the IR primitive layer.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::resolve_axis;
use crate::optim::resolver::{
    require_provable_static_int, resolve_int_or_bounded, BoundedInt, SiteKind,
};
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

// ── Concat / Stack helpers (shared) ──────────────────────────────────────

/// Recursive concatenation along an arbitrary axis. At axis 0 we splice the
/// outer lists; at axis k > 0 we recurse into each outer position with
/// axis k − 1. All input arrays must have matching shapes except along the
/// concatenation axis (caller is expected to validate).
pub(super) fn concat_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let mut all_values = Vec::new();
        let mut all_types = Vec::new();
        for arr in arrays {
            match arr {
                Value::List(d) | Value::Tuple(d) => {
                    all_values.extend(d.values.clone());
                    all_types.extend(d.elements_type.clone());
                }
                v => {
                    all_values.push(v.clone());
                    all_types.push(v.zinnia_type());
                }
            }
        }
        return Value::List(CompositeData { elements_type: all_types, values: all_values, value_id: ValueId::next() });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("concatenate: cannot apply axis > 0 to a 0-D value"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("concatenate: arrays must have matching ranks"),
            })
            .collect();
        rows.push(concat_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() })
}

pub fn np_concatenate(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len();
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array with {} dimensions",
            raw_axis, ndim
        );
    }

    let out = concat_recursive(&data.values, resolved as usize);
    let input_vids: Vec<ValueId> = data.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == data.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// Recursive stack along an arbitrary new axis. At axis 0 we wrap the input
/// arrays as the outer list directly. At axis k > 0 we walk the *first* axis
/// of the input arrays (which all have matching shapes) and recurse with
/// axis k − 1. The recursion bottoms out either in axis-0 wrap or in scalars.
fn stack_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let types = arrays.iter().map(|v| v.zinnia_type()).collect();
        return Value::List(CompositeData {
            elements_type: types,
            values: arrays.to_vec(),

            value_id: ValueId::next(),
        });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("stack: cannot stack at axis > input rank"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("stack: all input arrays must have the same rank"),
            })
            .collect();
        rows.push(stack_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() })
}

pub fn np_stack(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    // Stack inserts a new axis, so the result rank is input_rank + 1.
    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len() + 1;
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array of dimension {}",
            raw_axis,
            ndim - 1
        );
    }

    let out = stack_recursive(&data.values, resolved as usize);
    let input_vids: Vec<ValueId> = data.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == data.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

pub fn build_ndarray_from_flat(b: &mut IRBuilder, values: Vec<Value>, types: Vec<ZinniaType>, shape: &[usize]) -> Value {
    if shape.len() == 1 {
        Value::List(CompositeData { elements_type: types, values, value_id: ValueId::next() })
    } else {
        // Build nested structure
        let inner_size: usize = shape[1..].iter().product();
        let mut rows = Vec::new();
        for chunk in values.chunks(inner_size) {
            let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
            rows.push(build_ndarray_from_flat(b, chunk.to_vec(), chunk_types, &shape[1..]));
        }
        let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: row_types, values: rows, value_id: ValueId::next() })
    }
}

// ── NDArray helpers ───────────────────────────────────────────────

/// NDArray reshape: flatten, then rebuild with new shape.
pub fn ndarray_reshape(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let flat = crate::helpers::composite::flatten_composite(val);
    let total = flat.len();

    // Parse new shape from args — single tuple arg or multiple int args
    let new_shape: Vec<usize> = if args.len() == 1 {
        match &args[0] {
            Value::Tuple(data) | Value::List(data) => {
                data.values.iter().map(|v| {
                    let n = require_provable_static_int(b, v, SiteKind::ReshapeDim);
                    n as usize
                }).collect()
            }
            Value::Integer(_) => {
                let n = require_provable_static_int(b, &args[0], SiteKind::ReshapeDim);
                vec![n as usize]
            }
            _ => panic!("reshape: invalid shape argument"),
        }
    } else {
        args.iter().map(|v| {
            let n = require_provable_static_int(b, v, SiteKind::ReshapeDim);
            n as usize
        }).collect()
    };

    // Handle -1 (infer one dimension)
    let neg_count = new_shape.iter().filter(|&&s| s == usize::MAX).count();
    let final_shape: Vec<usize> = if neg_count == 1 {
        let known_product: usize = new_shape.iter().filter(|&&s| s != usize::MAX).product();
        assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
        new_shape.iter().map(|&s| if s == usize::MAX { total / known_product } else { s }).collect()
    } else {
        // Also handle -1 encoded as a large number from i64 cast
        let has_neg = args.iter().any(|a| a.int_val() == Some(-1));
        if has_neg {
            let known: Vec<usize> = new_shape.iter().copied().collect();
            let known_product: usize = known.iter().filter(|&&s| s < usize::MAX / 2).product();
            assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
            known.iter().map(|&s| if s >= usize::MAX / 2 { total / known_product } else { s }).collect()
        } else {
            new_shape
        }
    };

    let shape_product: usize = final_shape.iter().product();
    assert_eq!(total, shape_product, "reshape: total size mismatch ({} vs {})", total, shape_product);

    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    build_ndarray_from_flat(b, flat, types, &final_shape)
}

/// NDArray moveaxis: reorder axes by moving source axis to destination.
pub fn ndarray_moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "moveaxis: requires source and destination arguments");

    let src = {
        let s: i64 = require_provable_static_int(b, &args[0], SiteKind::Axis);
        if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
    };
    let dst = {
        let d: i64 = require_provable_static_int(b, &args[1], SiteKind::Axis);
        if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
    };
    assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

    // Build permutation: remove src, insert at dst
    let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
    order.insert(dst, src);

    let axes_val: Vec<Value> = order.iter()
        .map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None)))
        .collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; order.len()],
        values: axes_val,

        value_id: ValueId::next(),
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// NDArray repeat: repeat array elements along an axis.
pub fn ndarray_repeat(b: &mut IRBuilder, val: &Value, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let repeats_arg = args.first()
        .expect("repeat: repeats argument required");
    let axis = kwargs.get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    // Bounded-admission fast path: 1-D `val` with a scalar bounded `k` and
    // no axis (or axis=0) routes through zkRAM per-cell symbolic writes.
    // The output buffer has `len_arr * k_max` slots; slot `i` gets
    // `flat_src[(i / k) mod len_arr]`. Runtime mask via `runtime_length =
    // len_arr * k`. Multi-D input, tuple repeats, or axis != 0 fall
    // through to the static path.
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    if src_shape.len() == 1
        && matches!(repeats_arg, Value::Integer(_))
        && (axis.is_none() || axis == Some(0))
    {
        match resolve_int_or_bounded(b, repeats_arg, SiteKind::RepeatCount, None) {
            BoundedInt::Bounded { max, .. } => {
                let k_max = max.max(0) as usize;
                let flat_src = crate::helpers::composite::flatten_composite(val);
                let len_arr = flat_src.len();
                let dtype = match flat_src.first() {
                    Some(Value::Float(_)) => crate::types::NumberType::Float,
                    _ => crate::types::NumberType::Integer,
                };
                use crate::ops::dyn_ndarray::{
                    constructors::dyn_from_values_with_active, value_to_scalar_i64,
                };

                // Materialize source into a read-only segment for symbolic
                // reads. Length-zero input: nothing to repeat.
                if len_arr == 0 || k_max == 0 {
                    let runtime_length = b.ir_constant_int(0);
                    return dyn_from_values_with_active(
                        b,
                        Vec::new(),
                        runtime_length,
                        dtype,
                    );
                }
                let src_scalars: Vec<crate::types::ScalarValue<i64>> = flat_src
                    .iter()
                    .map(value_to_scalar_i64)
                    .collect();
                let src_seg = crate::helpers::segment::alloc_and_write(b, &src_scalars, dtype);

                // Output buffer pre-filled with defaults; per-cell symbolic
                // writes overwrite the active region. Slots beyond
                // runtime_length are never read by the subscript machinery.
                let default_sv = value_to_scalar_i64(&match dtype {
                    crate::types::NumberType::Float => b.ir_constant_float(0.0),
                    _ => b.ir_constant_int(0),
                });
                let init = vec![default_sv; len_arr * k_max];
                let out_seg = crate::helpers::segment::alloc_and_write(b, &init, dtype);

                let len_arr_const = b.ir_constant_int(len_arr as i64);
                let len_vid = len_arr_const.value_id().unwrap();
                for i in 0..(len_arr * k_max) {
                    let i_const = b.ir_constant_int(i as i64);
                    let arr_idx = b.ir_div_i(&i_const, repeats_arg);
                    let arr_idx_mod = b.ir_mod_i(&arr_idx, &len_arr_const);
                    let src_val = b.ir_read_memory(src_seg, &arr_idx_mod);
                    b.ir_write_memory(out_seg, &i_const, &src_val);
                }
                let runtime_length = b.ir_mul_i(&len_arr_const, repeats_arg);
                let runtime_length_sv = value_to_scalar_i64(&runtime_length);
                let runtime_length_vid = runtime_length_sv.value_id;

                // Fact: runtime_length == len_arr * k.
                let k_vid = repeats_arg.value_id().unwrap();
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                formals.insert("k".to_string(), k_vid);
                b.fire_contract("dyn_repeat", runtime_length_vid, &formals);

                let logical_shape = vec![len_arr * k_max];
                let envelope =
                    crate::types::Envelope::from_static_shape(&mut b.dim_table, &logical_shape);
                let result = Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    envelope,
                    dtype,
                    segment_id: out_seg,
                    meta: crate::types::DynArrayMeta {
                        logical_shape,
                        logical_offset: 0,
                        logical_strides: vec![1],
                        runtime_length: runtime_length_sv.clone(),
                        runtime_rank: crate::types::ScalarValue::new(Some(1), None),
                        runtime_shape: vec![runtime_length_sv],
                        runtime_strides: vec![crate::types::ScalarValue::new(Some(1), None)],
                        runtime_offset: crate::types::ScalarValue::new(Some(0), None),
                    },
                    value_id: crate::types::ValueId::next(),
                });
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
            BoundedInt::Static(_) | BoundedInt::Neither => {
                // Fall through to the static path below.
            }
        }
    }

    let repeats: i64 = require_provable_static_int(b, repeats_arg, SiteKind::RepeatCount);

    if let Some(ax) = axis {
        // Repeat along specific axis
        let shape = crate::helpers::composite::get_composite_shape(val);
        let ndim = shape.len();
        let ax = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };

        if ax == 0 {
            if let Value::List(data) | Value::Tuple(data) = val {
                let mut new_vals = Vec::new();
                for v in &data.values {
                    for _ in 0..repeats {
                        new_vals.push(v.clone());
                    }
                }
                let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                let result = Value::List(CompositeData { elements_type: types, values: new_vals, value_id: ValueId::next() });
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
        }
        // For other axes, transpose so target axis is first, repeat, transpose back
        let mut fwd: Vec<usize> = (0..ndim).collect();
        fwd.swap(0, ax);
        let fwd_vals: Vec<Value> = fwd.iter().map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None))).collect();
        let fwd_tuple = Value::Tuple(CompositeData { elements_type: vec![ZinniaType::Integer; ndim], values: fwd_vals, value_id: ValueId::next() });
        let transposed = crate::helpers::ndarray::ndarray_transpose(b, val, &[fwd_tuple.clone()]);
        let repeated = ndarray_repeat(b, &transposed, args, &HashMap::new());
        crate::helpers::ndarray::ndarray_transpose(b, &repeated, &[fwd_tuple])
    } else {
        // No axis: flatten, then repeat each element
        let flat = crate::helpers::composite::flatten_composite(val);
        let mut new_vals = Vec::new();
        for v in &flat {
            for _ in 0..repeats {
                new_vals.push(v.clone());
            }
        }
        let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
        let result = Value::List(CompositeData { elements_type: types, values: new_vals, value_id: ValueId::next() });
        if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
            crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
        }
        result
    }
}

/// NDArray filter: select elements where mask is true.
pub fn ndarray_filter(_b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let mask = args.first().expect("filter: requires a mask argument");
    let elements = crate::helpers::composite::flatten_composite(val);
    let mask_elements = crate::helpers::composite::flatten_composite(mask);
    assert_eq!(elements.len(), mask_elements.len(), "filter: array and mask must have same size");

    // For static arrays, we can build a filtered result at compile time
    // by using select chains. The result length depends on the mask values.
    // If mask values are all statically known, produce a fixed-size result.
    let mut static_result = Vec::new();
    let mut all_static = true;
    for (elem, m) in elements.iter().zip(mask_elements.iter()) {
        match m.int_val().or_else(|| if matches!(m, Value::Boolean(bv) if bv.static_val == Some(true)) { Some(1) } else if matches!(m, Value::Boolean(bv) if bv.static_val == Some(false)) { Some(0) } else { None }) {
            Some(v) if v != 0 => static_result.push(elem.clone()),
            Some(_) => {} // masked out
            None => { all_static = false; break; }
        }
    }

    if all_static {
        let types = static_result.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: types, values: static_result, value_id: ValueId::next() })
    } else {
        panic!("filter: dynamic masks require DynamicNDArray (not yet supported in Rust backend)");
    }
}

pub fn ndarray_argmax_argmin_with_axis(b: &mut IRBuilder, val: &Value, axis: i64, is_max: bool) -> Value {
    if let Value::List(outer) | Value::Tuple(outer) = val {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

        if axis == 0 {
            // argmax along axis 0: for each column, find row with max/min
            if let Some(Value::List(first_row) | Value::Tuple(first_row)) = outer.values.first() {
                let ncols = first_row.values.len();
                let mut results = Vec::new();
                for col in 0..ncols {
                    let mut best_idx = b.ir_constant_int(0);
                    let mut best_val_opt: Option<Value> = None;
                    for (row_idx, row) in outer.values.iter().enumerate() {
                        if let Value::List(rd) | Value::Tuple(rd) = row {
                            if col < rd.values.len() {
                                if let Some(ref best_val) = best_val_opt {
                                    let cond = if is_max {
                                        b.ir_greater_than_i(&rd.values[col], best_val)
                                    } else {
                                        b.ir_less_than_i(&rd.values[col], best_val)
                                    };
                                    let idx_val = b.ir_constant_int(row_idx as i64);
                                    best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
                                    best_val_opt = Some(b.ir_select_i(&cond, &rd.values[col], best_val));
                                } else {
                                    best_val_opt = Some(rd.values[col].clone());
                                }
                            }
                        }
                    }
                    results.push(best_idx);
                }
                let types = vec![ZinniaType::Integer; results.len()];
                return Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() });
            }
        } else if axis == 1 {
            // argmax along axis 1: for each row, find column index of max/min
            let mut results = Vec::new();
            for row in &outer.values {
                results.push(crate::helpers::ndarray::ndarray_argmax_argmin(b, row, &[], is_max));
            }
            let types = vec![ZinniaType::Integer; results.len()];
            return Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() });
        }
    }
    crate::helpers::ndarray::ndarray_argmax_argmin(b, val, &[], is_max)
}


pub fn ndarray_shape(val: &Value) -> Value {
    // Return the shape as a tuple of constants
    match val {
        Value::List(data) => {
            // For a list, shape is (len,)
            let len_val = Value::Integer(crate::types::ScalarValue::new(Some(data.values.len() as i64), None));
            Value::Tuple(CompositeData {
                elements_type: vec![ZinniaType::Integer],
                values: vec![len_val],

                value_id: ValueId::next(),
            })
        }
        Value::NDArray(nd) => {
            let shape_vals: Vec<Value> = nd.shape.iter()
                .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                .collect();
            let types = shape_vals.iter().map(|_| ZinniaType::Integer).collect();
            Value::Tuple(CompositeData {
                elements_type: types,
                values: shape_vals,

                value_id: ValueId::next(),
            })
        }
        _ => Value::None,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Shape-manipulation helpers (swapaxes / flip / squeeze / expand_dims /
// broadcast_to / atleast_Nd / tile / vstack / hstack / dstack /
// column_stack / row_stack)
// ────────────────────────────────────────────────────────────────────────

/// Build a constant Integer Value from a usize.
fn const_int(b: &mut IRBuilder, n: usize) -> Value {
    b.ir_constant_int(n as i64)
}

/// `np.swapaxes(arr, a1, a2)` — swap two axes.
pub fn ndarray_swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "swapaxes: requires two axis arguments");
    let a1 = resolve_axis(
        require_provable_static_int(b, &args[0], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let a2 = resolve_axis(
        require_provable_static_int(b, &args[1], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let mut order: Vec<usize> = (0..ndim).collect();
    order.swap(a1, a2);
    let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; ndim],
        values: axes_vals,

        value_id: ValueId::next(),
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// Reverse `val`'s elements along axis `axis`. Other axes keep order.
fn flip_along(val: &Value, axis: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values.iter().rev().cloned().collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,

                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> =
                    d.values.iter().map(|v| flip_along(v, axis - 1)).collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,

                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    }
}

/// `np.flip(arr, axis=None)` — reverse along the given axis (or all axes
/// when no axis is specified).
pub fn np_flip(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("flip: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));
    let axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => (0..ndim).collect(),
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                resolve_axis(n, ndim, "flip")
            })
            .collect(),
        Some(a) => {
            let n: i64 = require_provable_static_int(b, a, SiteKind::Axis);
            vec![resolve_axis(n, ndim, "flip")]
        }
    };
    let mut out = val.clone();
    for ax in axes {
        out = flip_along(&out, ax);
    }
    out
}

/// `np.flipud(arr)` — flip along axis 0.
pub fn np_flipud(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("flipud: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.is_empty() {
        panic!("flipud: input must be at least 1-D");
    }
    flip_along(val, 0)
}

/// `np.fliplr(arr)` — flip along axis 1 (requires ndim ≥ 2).
pub fn np_fliplr(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("fliplr: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("fliplr: input must be at least 2-D");
    }
    flip_along(val, 1)
}

/// `np.rot90(arr, k=1, axes=(0, 1))` — rotate 90° counter-clockwise k times
/// in the plane spanned by `axes`. Each k=1 rotation is `flip(axes[1])`
/// then `swapaxes(axes[0], axes[1])`, matching NumPy's reference.
pub fn np_rot90(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("rot90: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim < 2 {
        panic!("rot90: input must be at least 2-D");
    }
    let k = kwargs
        .get("k")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(1);
    let axes_arg = kwargs.get("axes").or_else(|| args.get(2));
    let (a0, a1) = match axes_arg {
        Some(Value::Tuple(d)) | Some(Value::List(d)) if d.values.len() == 2 => {
            let a = resolve_axis(d.values[0].int_val().unwrap_or(0), ndim, "rot90");
            let bb = resolve_axis(d.values[1].int_val().unwrap_or(1), ndim, "rot90");
            (a, bb)
        }
        _ => (0usize, 1usize),
    };
    if a0 == a1 {
        panic!("rot90: axes must be different");
    }
    let k = ((k % 4) + 4) % 4;
    let mut out = val.clone();
    for _ in 0..k {
        out = flip_along(&out, a1);
        let mut order: Vec<usize> = (0..ndim).collect();
        order.swap(a0, a1);
        let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
        let axes_tuple = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer; ndim],
            values: axes_vals,

            value_id: ValueId::next(),
        });
        out = crate::helpers::ndarray::ndarray_transpose(b, &out, &[axes_tuple]);
    }
    out
}

/// `np.squeeze(arr, axis=None)` — drop axes of length 1.
pub fn np_squeeze(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("squeeze: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));

    let target_axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
            .collect(),
        Some(Value::Tuple(d)) | Some(Value::List(d)) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                resolve_axis(n, ndim, "squeeze")
            })
            .collect(),
        Some(a) => {
            let n: i64 = require_provable_static_int(b, a, SiteKind::Axis);
            vec![resolve_axis(n, ndim, "squeeze")]
        }
    };
    for &ax in &target_axes {
        if shape[ax] != 1 {
            panic!(
                "squeeze: cannot select an axis to squeeze out which has size not equal to one (axis {})",
                ax
            );
        }
    }
    if target_axes.is_empty() {
        return val.clone();
    }
    let new_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if target_axes.contains(&i) { None } else { Some(d) })
        .collect();
    let flat = crate::helpers::composite::flatten_composite(val);
    if new_shape.is_empty() {
        let out = flat.into_iter().next().unwrap_or(Value::None);
        if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
            crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
        }
        return out;
    }
    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    let out = crate::helpers::composite::build_nested_value(flat, types, &new_shape);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.expand_dims(arr, axis)` — insert a new axis of length 1 at `axis`.
pub fn np_expand_dims(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("expand_dims: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_arg = args
        .get(1)
        .expect("expand_dims: axis argument required");
    let axis: i64 = require_provable_static_int(b, axis_arg, SiteKind::NewAxisPosition);
    let new_ndim = ndim + 1;
    let resolved = if axis < 0 { new_ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= new_ndim as i64 {
        panic!(
            "expand_dims: axis {} is out of bounds for array of rank {}",
            axis, new_ndim
        );
    }
    let pos = resolved as usize;
    fn insert_at(val: &Value, pos: usize) -> Value {
        if pos == 0 {
            Value::List(CompositeData {
                elements_type: vec![val.zinnia_type()],
                values: vec![val.clone()],

                value_id: ValueId::next(),
            })
        } else {
            match val {
                Value::List(d) | Value::Tuple(d) => {
                    let new_vals: Vec<Value> =
                        d.values.iter().map(|v| insert_at(v, pos - 1)).collect();
                    let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData {
                        elements_type: new_types,
                        values: new_vals,

                        value_id: ValueId::next(),
                    })
                }
                _ => val.clone(),
            }
        }
    }
    let out = insert_at(val, pos);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.broadcast_to(arr, shape)` — materialize the broadcast to a target
/// shape. Thin wrapper around the broadcasting helper.
pub fn np_broadcast_to(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("broadcast_to: requires an array argument");
    let shape_arg = args.get(1).expect("broadcast_to: requires a shape argument");
    let target: Vec<usize> = match shape_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::ShapeAxis(i));
                n as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n: i64 = require_provable_static_int(b, shape_arg, SiteKind::ShapeAxis(0));
            vec![n as usize]
        }
        _ => panic!("broadcast_to: invalid shape argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    match crate::helpers::broadcast::broadcast_shapes(&src_shape, &target) {
        Some(s) if s == target => {}
        _ => panic!(
            "broadcast_to: shape {:?} cannot be broadcast to {:?}",
            src_shape, target
        ),
    }
    // broadcast_to is sound for relay: it replicates existing cells, so every
    // new element equals an existing one — `forall_eq_const(in, k)` ⇒
    // `forall_eq_const(out, k)`.
    let out = crate::helpers::broadcast::materialize_to_shape(val, &target);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.atleast_1d/2d/3d(arr)` — prepend unit axes until rank ≥ n.
pub fn np_atleast_nd(_b: &mut IRBuilder, args: &[Value], n: usize) -> Value {
    let val = args.first().expect("atleast_Nd: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() >= n {
        return val.clone();
    }
    let mut out = val.clone();
    for _ in 0..(n - shape.len()) {
        out = Value::List(CompositeData {
            elements_type: vec![out.zinnia_type()],
            values: vec![out],

            value_id: ValueId::next(),
        });
    }
    out
}

/// `np.tile(arr, reps)` — repeat `arr` according to `reps`.
pub fn np_tile(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("tile: requires an array argument");
    let reps_arg = args.get(1).expect("tile: requires a reps argument");

    // Bounded-admission fast path: 1-D `arr` with a scalar bounded `reps`
    // promotes to a `DynamicNDArray` via the natural-pad approach.
    // `tile([a,b,c], k)` for `k ∈ [0, k_max]` is a prefix of the fully
    // expanded `tile([a,b,c], k_max)`; padding to `k_max` and truncating
    // via `runtime_length = len_arr * k` gives the right user-visible
    // output. (Multi-D bounded tile is out of scope.)
    if matches!(reps_arg, Value::Integer(_))
        && crate::helpers::composite::get_composite_shape(val).len() == 1
    {
        match resolve_int_or_bounded(b, reps_arg, SiteKind::RepeatCount, None) {
            BoundedInt::Bounded { max, .. } => {
                let k_max = max.max(0) as usize;
                let flat_src = crate::helpers::composite::flatten_composite(val);
                let len_arr = flat_src.len();
                let dtype = match flat_src.first() {
                    Some(Value::Float(_)) => crate::types::NumberType::Float,
                    _ => crate::types::NumberType::Integer,
                };
                use crate::ops::dyn_ndarray::{
                    constructors::dyn_from_values_with_active, value_to_scalar_i64,
                };
                let mut values: Vec<crate::types::ScalarValue<i64>> =
                    Vec::with_capacity(len_arr * k_max);
                for _ in 0..k_max {
                    for v in &flat_src {
                        values.push(value_to_scalar_i64(v));
                    }
                }
                let len_const = b.ir_constant_int(len_arr as i64);
                let len_vid = len_const.value_id().unwrap();
                let runtime_length = b.ir_mul_i(&len_const, reps_arg);
                let result = dyn_from_values_with_active(b, values, runtime_length, dtype);
                // Fact: runtime_length == len_arr * k.
                let runtime_length_vid = match &result {
                    Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
                    _ => unreachable!(),
                };
                let k_vid = reps_arg.value_id().unwrap();
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                formals.insert("k".to_string(), k_vid);
                b.fire_contract("dyn_tile", runtime_length_vid, &formals);
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
            BoundedInt::Static(_) | BoundedInt::Neither => {
                // Fall through to the static path below, which will
                // either succeed (Static) or panic with the standard
                // RepeatCount diagnostic (Neither).
            }
        }
    }

    let reps: Vec<usize> = match reps_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::RepeatCount);
                n.max(0) as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n: i64 = require_provable_static_int(b, reps_arg, SiteKind::RepeatCount);
            vec![n.max(0) as usize]
        }
        _ => panic!("tile: invalid reps argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    let rank = src_shape.len().max(reps.len());
    let mut padded_shape = vec![1usize; rank - src_shape.len()];
    padded_shape.extend_from_slice(&src_shape);
    let mut padded_reps = vec![1usize; rank - reps.len()];
    padded_reps.extend_from_slice(&reps);

    // Promote val to padded rank by prepending unit axes if needed.
    let mut promoted = val.clone();
    for _ in 0..(rank - src_shape.len()) {
        promoted = Value::List(CompositeData {
            elements_type: vec![promoted.zinnia_type()],
            values: vec![promoted],

            value_id: ValueId::next(),
        });
    }

    let target_shape: Vec<usize> = padded_shape
        .iter()
        .zip(padded_reps.iter())
        .map(|(s, r)| s * r)
        .collect();

    let total: usize = target_shape.iter().product();
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * padded_shape[i + 1];
    }
    let flat_src = crate::helpers::composite::flatten_composite(&promoted);
    let mut out_flat: Vec<Value> = Vec::with_capacity(total);
    for out_idx in 0..total {
        let mut remainder = out_idx;
        let mut src_flat = 0usize;
        for d in 0..rank {
            let coord = remainder / out_strides[d];
            remainder %= out_strides[d];
            let src_coord = coord % padded_shape[d];
            src_flat += src_coord * src_strides[d];
        }
        out_flat.push(flat_src[src_flat].clone());
    }
    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    let out = crate::helpers::composite::build_nested_value(out_flat, types, &target_shape);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

// ── stack convenience wrappers ─────────────────────────────────────────

/// Promote a 1-D array to a 2-D row (`(N,)` → `(1, N)`); leave higher-rank
/// arrays untouched. Used by vstack/row_stack.
fn promote_to_row(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        Value::List(CompositeData {
            elements_type: vec![val.zinnia_type()],
            values: vec![val.clone()],

            value_id: ValueId::next(),
        })
    } else {
        val.clone()
    }
}

/// Promote a 1-D array to a 2-D column (`(N,)` → `(N, 1)`).
fn promote_to_column(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() == 1 {
        if let Value::List(d) | Value::Tuple(d) = val {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| {
                    Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],

                        value_id: ValueId::next(),
                    })
                })
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: new_vals,

                value_id: ValueId::next(),
            });
        }
    }
    val.clone()
}

/// `np.vstack(arrays)` — stack along axis 0, promoting 1-D inputs to rows.
pub fn np_vstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_row).collect();
    let out = concat_recursive(&promoted, 0);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.hstack(arrays)` — concatenate along axis 1 for ≥2-D inputs, or along
/// axis 0 for 1-D inputs (NumPy convention).
pub fn np_hstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let any_multi = arrays
        .values
        .iter()
        .any(|v| crate::helpers::composite::get_composite_shape(v).len() >= 2);
    let axis = if any_multi { 1 } else { 0 };
    let out = concat_recursive(&arrays.values, axis);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.dstack(arrays)` — stack along axis 2, promoting lower-rank inputs.
pub fn np_dstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let raw_values = arrays.values.clone();
    let promoted: Vec<Value> = raw_values
        .iter()
        .map(|v| {
            let shape = crate::helpers::composite::get_composite_shape(v);
            match shape.len() {
                1 => {
                    // (N,) -> (1, N) -> (1, N, 1)
                    let row = Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],

                        value_id: ValueId::next(),
                    });
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[row, two])
                }
                2 => {
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[v.clone(), two])
                }
                _ => v.clone(),
            }
        })
        .collect();
    let out = concat_recursive(&promoted, 2);
    let input_vids: Vec<ValueId> = raw_values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == raw_values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.column_stack(arrays)` — 1-D arrays become columns of a 2-D output.
pub fn np_column_stack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_column).collect();
    let out = concat_recursive(&promoted, 1);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.row_stack(arrays)` — alias of vstack.
pub fn np_row_stack(b: &mut IRBuilder, args: &[Value]) -> Value {
    np_vstack(b, args)
}

// ────────────────────────────────────────────────────────────────────────
// Splitting family (split / array_split / hsplit / vsplit / dsplit)
// ────────────────────────────────────────────────────────────────────────

/// Take the slice `start..stop` of `val` along axis `axis`. Other axes are
/// kept intact. Used by all the splitting helpers below.
fn slice_along_axis(val: &Value, axis: usize, start: usize, stop: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values[start..stop].to_vec();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,

                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d
                    .values
                    .iter()
                    .map(|v| slice_along_axis(v, axis - 1, start, stop))
                    .collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,

                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    }
}

/// Compute the section boundaries (cumulative sizes) for `np.split` and
/// `np.array_split`. For `array_split`, when `n` does not evenly divide
/// `length`, the first `length % n` sections get one extra element each
/// (matching NumPy).
fn compute_split_boundaries(
    b: &mut IRBuilder,
    length: usize,
    sections: &Value,
    allow_uneven: bool,
) -> Vec<(usize, usize)> {
    match sections {
        Value::Integer(_) => {
            let n = require_provable_static_int(b, sections, SiteKind::SplitSections);
            let n = n as usize;
            if n == 0 {
                panic!("split: number of sections must be > 0");
            }
            if !allow_uneven && length % n != 0 {
                panic!(
                    "split: array of length {} cannot be split into {} equal sections",
                    length, n
                );
            }
            let base = length / n;
            let extras = length % n;
            let mut out = Vec::with_capacity(n);
            let mut cursor = 0usize;
            for i in 0..n {
                let sz = base + if i < extras { 1 } else { 0 };
                out.push((cursor, cursor + sz));
                cursor += sz;
            }
            out
        }
        Value::List(d) | Value::Tuple(d) => {
            // Index list: split *at* these indices.
            let mut indices: Vec<usize> = d
                .values
                .iter()
                .map(|v| {
                    let i = require_provable_static_int(b, v, SiteKind::SplitSections);
                    i.max(0).min(length as i64) as usize
                })
                .collect();
            indices.push(length);
            let mut out = Vec::with_capacity(indices.len());
            let mut prev = 0usize;
            for &i in &indices {
                out.push((prev, i.max(prev)));
                prev = i.max(prev);
            }
            out
        }
        _ => panic!("split: sections must be an int or a list of indices"),
    }
}

/// Shared body for `np.split` / `np.array_split` along an explicit axis.
fn split_impl(
    b: &mut IRBuilder,
    val: &Value,
    sections: &Value,
    axis: i64,
    allow_uneven: bool,
    op: &str,
) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim == 0 {
        panic!("{}: cannot split a 0-D value", op);
    }
    let ax = resolve_axis(axis, ndim, op);
    let length = shape[ax];
    let bounds = compute_split_boundaries(b, length, sections, allow_uneven);
    let parts: Vec<Value> = bounds
        .into_iter()
        .map(|(s, e)| slice_along_axis(val, ax, s, e))
        .collect();
    let types = parts.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values: parts,

        value_id: ValueId::next(),
    })
}

/// `np.split(arr, sections, axis=0)` — equal-section split (errors if the
/// sections don't divide evenly).
pub fn np_split(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("split: requires an array argument");
    let sections = args.get(1).expect("split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(b, val, sections, axis, false, "split")
}

/// `np.array_split(arr, sections, axis=0)` — like split but allows uneven
/// sections; the first `length % n` sections get one extra element.
pub fn np_array_split(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let val = args.first().expect("array_split: requires an array argument");
    let sections = args.get(1).expect("array_split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(b, val, sections, axis, true, "array_split")
}

/// `np.hsplit(arr, sections)` — split along axis 1 for ≥2-D, axis 0 for 1-D.
pub fn np_hsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("hsplit: requires an array argument");
    let sections = args.get(1).expect("hsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let axis = if shape.len() >= 2 { 1 } else { 0 };
    split_impl(b, val, sections, axis, false, "hsplit")
}

/// `np.vsplit(arr, sections)` — split along axis 0 (requires ≥2-D).
pub fn np_vsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("vsplit: requires an array argument");
    let sections = args.get(1).expect("vsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("vsplit: input must be at least 2-D");
    }
    split_impl(b, val, sections, 0, false, "vsplit")
}

/// `np.dsplit(arr, sections)` — split along axis 2 (requires ≥3-D).
pub fn np_dsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("dsplit: requires an array argument");
    let sections = args.get(1).expect("dsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 3 {
        panic!("dsplit: input must be at least 3-D");
    }
    split_impl(b, val, sections, 2, false, "dsplit")
}

// ────────────────────────────────────────────────────────────────────────
// np.block — recursive nested concatenation
// ────────────────────────────────────────────────────────────────────────

/// Recursive worker for `np.block`. At each block level (going from the
/// outermost level inward), we recurse on each child with `block_depth − 1`,
/// then concat the results along the appropriate axis.
///
/// The axis follows NumPy's "negative axis from the result rank" rule: at
/// the outermost level we concat along axis `result_ndim − block_depth`; at
/// the innermost block level we concat along axis `result_ndim − 1`.
fn block_recursive(val: &Value, block_depth: usize, result_ndim: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }
    let children: Vec<Value> = match val {
        Value::List(d) | Value::Tuple(d) => d
            .values
            .iter()
            .map(|c| block_recursive(c, block_depth - 1, result_ndim))
            .collect(),
        _ => return val.clone(),
    };
    let axis = result_ndim - block_depth;
    concat_recursive(&children, axis)
}

/// Walk every leaf in the nested block structure at the given depth. Used to
/// validate that all leaves share a rank.
fn collect_block_leaves(val: &Value, block_depth: usize) -> Vec<Value> {
    if block_depth == 0 {
        return vec![val.clone()];
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let mut all = Vec::new();
            for child in &d.values {
                all.extend(collect_block_leaves(child, block_depth - 1));
            }
            all
        }
        _ => vec![val.clone()],
    }
}

/// `np.block(arrays, block_depth)` — recursive nested concatenation. The
/// block depth must be supplied by the caller (typically computed from the
/// AST nesting in `ir_gen/named_attr.rs`, since after a Python list literal
/// has been visited into a `Value::List` we can no longer distinguish "a
/// nested block of arrays" from "a single high-rank ndarray").
///
/// All leaf arrays must share the same rank, and that rank must be ≥
/// `block_depth`. NumPy auto-promotes mixed-rank leaves via `atleast_Nd`;
/// that case is currently out of scope and produces a hard error.
pub fn np_block_with_depth(val: &Value, block_depth: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }

    let leaves = collect_block_leaves(val, block_depth);
    if leaves.is_empty() {
        return val.clone();
    }
    let first_rank = crate::helpers::composite::get_composite_shape(&leaves[0]).len();
    for leaf in &leaves {
        let r = crate::helpers::composite::get_composite_shape(leaf).len();
        if r != first_rank {
            panic!(
                "block: all leaf arrays must currently have the same rank \
                 (got {} and {}). Mixed-rank block (NumPy auto-promotes via \
                 atleast_Nd) is not yet supported on static ndarrays.",
                first_rank, r
            );
        }
    }
    if first_rank < block_depth {
        panic!(
            "block: leaf arrays of rank {} cannot be combined into a block \
             of nesting depth {}. Promote them with np.atleast_Nd first, or \
             use np.stack / np.concatenate directly.",
            first_rank, block_depth
        );
    }

    let result_ndim = first_rank.max(block_depth);
    block_recursive(val, block_depth, result_ndim)
}
