//! Free-function versions of IRBuilder value operations.
//!
//! These functions implement binary operations, conditional selects,
//! dynamic list indexing, and other value-level operations, taking
//! `&mut IRBuilder` as an explicit parameter instead of `&mut self`.

use crate::builder::IRBuilder;
use crate::types::{CompositeData, Value, ZinniaType};

/// Conditional select: if cond { tv } else { fv }, with element-wise support.
pub fn select_value(b: &mut IRBuilder, cond: &Value, tv: &Value, fv: &Value) -> Value {
    match (tv, fv) {
        (Value::List(td), Value::List(fd)) if td.values.len() == fd.values.len() => {
            let results: Vec<Value> = td.values.iter().zip(fd.values.iter())
                .map(|(t, f)| select_value(b, cond, t, f))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results })
        }
        (Value::Tuple(td), Value::Tuple(fd)) if td.values.len() == fd.values.len() => {
            let results: Vec<Value> = td.values.iter().zip(fd.values.iter())
                .map(|(t, f)| select_value(b, cond, t, f))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::Tuple(CompositeData { elements_type: types, values: results })
        }
        // If types don't match (e.g., list vs scalar), just use the true value
        // (can't do conditional select across different structures)
        (Value::List(_) | Value::Tuple(_), _) | (_, Value::List(_) | Value::Tuple(_)) => {
            tv.clone()
        }
        _ => b.ir_select_i(cond, tv, fv),
    }
}

/// Convert a value to a scalar boolean, reducing composites via AND.
pub fn to_scalar_bool(b: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            if data.values.is_empty() {
                return b.ir_constant_bool(true);
            }
            let mut acc = to_scalar_bool(b, &data.values[0]);
            for elem in &data.values[1..] {
                let elem_bool = to_scalar_bool(b, elem);
                acc = b.ir_logical_and(&acc, &elem_bool);
            }
            acc
        }
        _ => b.ir_bool_cast(val),
    }
}

/// Apply a binary operation, with element-wise support for composite types.
pub fn apply_binary_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    // List/tuple concatenation via `+` (only for different-length composites
    // or when both are pure integer lists — same-length composites do element-wise)
    if op == "add" {
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd)) | (Value::Tuple(ld), Value::List(rd))
            | (Value::List(ld), Value::Tuple(rd)) | (Value::Tuple(ld), Value::Tuple(rd)) => {
                // Same-length composites: element-wise addition (ndarray behavior)
                if ld.values.len() == rd.values.len() && !ld.values.is_empty() {
                    let results: Vec<Value> = ld.values.iter().zip(rd.values.iter())
                        .map(|(l, r)| apply_binary_op(b, "add", l, r))
                        .collect();
                    let types = results.iter().map(|v| v.zinnia_type()).collect();
                    return Value::List(CompositeData { elements_type: types, values: results });
                }
                // Different-length composites: by default this is Python
                // list concatenation. But if both operands look like
                // ndarrays (purely numeric leaves) AND their shapes are
                // broadcast-compatible, the user means NumPy element-wise
                // addition, not concatenation. In that case fall out of
                // this branch and let the shared broadcasting arm further
                // down handle it.
                let lshape = crate::helpers::composite::get_composite_shape(lhs);
                let rshape = crate::helpers::composite::get_composite_shape(rhs);
                let both_numeric = is_numeric_composite(lhs) && is_numeric_composite(rhs);
                if !(both_numeric
                    && crate::helpers::broadcast::broadcast_shapes(&lshape, &rshape).is_some())
                {
                    let mut values = ld.values.clone();
                    values.extend(rd.values.clone());
                    let types = values.iter().map(|v| v.zinnia_type()).collect();
                    let is_tuple = matches!(lhs, Value::Tuple(_));
                    return if is_tuple {
                        Value::Tuple(CompositeData { elements_type: types, values })
                    } else {
                        Value::List(CompositeData { elements_type: types, values })
                    };
                }
            }
            _ => {}
        }
    }
    // List/tuple repetition via `*`
    if op == "mul" {
        match (lhs, rhs) {
            (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.int_val().is_some() => {
                let n = rhs.int_val().unwrap().max(0) as usize;
                let mut values = Vec::new();
                for _ in 0..n {
                    values.extend(ld.values.clone());
                }
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                let is_tuple = matches!(lhs, Value::Tuple(_));
                return if is_tuple {
                    Value::Tuple(CompositeData { elements_type: types, values })
                } else {
                    Value::List(CompositeData { elements_type: types, values })
                };
            }
            (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.int_val().is_some() => {
                let n = lhs.int_val().unwrap().max(0) as usize;
                let mut values = Vec::new();
                for _ in 0..n {
                    values.extend(rd.values.clone());
                }
                let types = values.iter().map(|v| v.zinnia_type()).collect();
                let is_tuple = matches!(rhs, Value::Tuple(_));
                return if is_tuple {
                    Value::Tuple(CompositeData { elements_type: types, values })
                } else {
                    Value::List(CompositeData { elements_type: types, values })
                };
            }
            _ => {}
        }
    }
    // Composite comparison: handle eq/ne/lt/lte/gt/gte for composites.
    //
    // NumPy-style element-wise comparison fires when both sides are
    // *numeric* composites (purely numeric leaves) with broadcast-compatible
    // shapes — in that case the result is an ndarray of booleans, not a
    // single scalar bool. This is the same heuristic used by the binary
    // arithmetic broadcast path so the two stay consistent.
    //
    // For non-numeric composites (heterogeneous Python lists, lists of
    // strings, tuples of mixed types, etc.) we keep the existing
    // lexicographic-and-reduce semantics handled by `composite_comparison`,
    // which is closer to Python's `list == list` behaviour.
    if matches!(op, "eq" | "ne" | "lt" | "lte" | "gt" | "gte") {
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd))
            | (Value::Tuple(ld), Value::List(rd))
            | (Value::List(ld), Value::Tuple(rd))
            | (Value::Tuple(ld), Value::Tuple(rd)) => {
                if is_numeric_composite(lhs) && is_numeric_composite(rhs) {
                    let lshape = crate::helpers::composite::get_composite_shape(lhs);
                    let rshape = crate::helpers::composite::get_composite_shape(rhs);
                    if let Some(out_shape) =
                        crate::helpers::broadcast::broadcast_shapes(&lshape, &rshape)
                    {
                        let l = if lshape == out_shape {
                            lhs.clone()
                        } else {
                            crate::helpers::broadcast::materialize_to_shape(lhs, &out_shape)
                        };
                        let r = if rshape == out_shape {
                            rhs.clone()
                        } else {
                            crate::helpers::broadcast::materialize_to_shape(rhs, &out_shape)
                        };
                        // Walk element-wise directly. We can't just recurse
                        // back through `apply_binary_op` on the whole
                        // composites here, because when the shapes already
                        // match the broadcast is a no-op and we'd re-enter
                        // this same branch forever.
                        let (l_data, r_data) = match (&l, &r) {
                            (Value::List(a), Value::List(b))
                            | (Value::List(a), Value::Tuple(b))
                            | (Value::Tuple(a), Value::List(b))
                            | (Value::Tuple(a), Value::Tuple(b)) => (a.clone(), b.clone()),
                            _ => unreachable!("numeric composites guaranteed by check above"),
                        };
                        let results: Vec<Value> = l_data
                            .values
                            .iter()
                            .zip(r_data.values.iter())
                            .map(|(lv, rv)| apply_binary_op(b, op, lv, rv))
                            .collect();
                        let types = results.iter().map(|v| v.zinnia_type()).collect();
                        return Value::List(CompositeData {
                            elements_type: types,
                            values: results,
                        });
                    }
                }
                return composite_comparison(b, op, ld, rd);
            }
            _ => {}
        }
    }
    // Element-wise: both composite with matching length (for arithmetic ops)
    match (lhs, rhs) {
        (Value::List(ld), Value::List(rd)) | (Value::Tuple(ld), Value::List(rd))
        | (Value::List(ld), Value::Tuple(rd)) | (Value::Tuple(ld), Value::Tuple(rd))
            if ld.values.len() == rd.values.len() =>
        {
            let results: Vec<Value> = ld.values.iter().zip(rd.values.iter())
                .map(|(l, r)| apply_binary_op(b, op, l, r))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: results });
        }
        // Shape-level broadcasting: both composites, mismatched shapes that are
        // broadcast-compatible (NumPy semantics). Materialize both operands to
        // the broadcast shape, then recurse — the equal-length arm above will
        // handle the actual element-wise op.
        //
        // This deliberately fires only for *arithmetic* ops on composite ↔
        // composite. Pure list/tuple `+` concatenation and `*` repetition were
        // already handled near the top of this function, so by the time we get
        // here those cases are out of scope. Composite comparison (`eq`/`lt`/…)
        // is dispatched even earlier and is intentionally left untouched —
        // Python lexicographic comparison semantics are preserved.
        // `mat_mul` is excluded — matmul has its own shape rules and error
        // reporting in `static_ndarray_ops::matmul`, reached via the scalar
        // fallback below.
        (Value::List(_) | Value::Tuple(_), Value::List(_) | Value::Tuple(_)) if op != "mat_mul" => {
            let lshape = crate::helpers::composite::get_composite_shape(lhs);
            let rshape = crate::helpers::composite::get_composite_shape(rhs);
            if let Some(out_shape) = crate::helpers::broadcast::broadcast_shapes(&lshape, &rshape) {
                if out_shape != lshape || out_shape != rshape {
                    let l_mat = crate::helpers::broadcast::materialize_to_shape(lhs, &out_shape);
                    let r_mat = crate::helpers::broadcast::materialize_to_shape(rhs, &out_shape);
                    return apply_binary_op(b, op, &l_mat, &r_mat);
                }
                // Same shape but different lengths? Shouldn't happen — fall through.
            } else {
                panic!(
                    "operands could not be broadcast together with shapes {:?} {:?}",
                    lshape, rshape
                );
            }
        }
        // Broadcasting: scalar op composite
        (_, Value::List(rd)) | (_, Value::Tuple(rd)) if lhs.is_number() => {
            let results: Vec<Value> = rd.values.iter()
                .map(|r| apply_binary_op(b, op, lhs, r))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: results });
        }
        (Value::List(ld), _) | (Value::Tuple(ld), _) if rhs.is_number() => {
            let results: Vec<Value> = ld.values.iter()
                .map(|l| apply_binary_op(b, op, l, rhs))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: results });
        }
        _ => {}
    }
    // Class (type) comparison
    if matches!((lhs, rhs), (Value::Class(_), Value::Class(_))) {
        if let (Value::Class(lt), Value::Class(rt)) = (lhs, rhs) {
            let types_equal = lt == rt;
            return match op {
                "eq" => b.ir_constant_bool(types_equal),
                "ne" => b.ir_constant_bool(!types_equal),
                _ => b.ir_constant_bool(false),
            };
        }
    }
    // Scalar operation
    apply_scalar_binary_op(b, op, lhs, rhs)
}

pub fn apply_scalar_binary_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    let use_float = matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_));
    if use_float {
        // Implicit cast: ensure both operands are float
        let lf = ensure_float(b, lhs);
        let rf = ensure_float(b, rhs);
        // Float operations
        match op {
            "add" => b.ir_add_f(&lf, &rf),
            "sub" => b.ir_sub_f(&lf, &rf),
            "mul" => b.ir_mul_f(&lf, &rf),
            "div" => b.ir_div_f(&lf, &rf),
            "pow" => b.ir_pow_f(&lf, &rf),
            "mod" | "floor_div" => {
                // Fallback to integer ops for these
                b.ir_mod_i(lhs, rhs)
            }
            "eq" => b.ir_equal_f(&lf, &rf),
            "ne" => {
                let eq = b.ir_equal_f(&lf, &rf);
                b.ir_logical_not(&eq)
            }
            "lt" => b.ir_less_than_f(&lf, &rf),
            "lte" => b.ir_less_than_or_equal_f(&lf, &rf),
            "gt" => b.ir_greater_than_f(&lf, &rf),
            "gte" => b.ir_greater_than_or_equal_f(&lf, &rf),
            "and" => b.ir_logical_and(&lf, &rf),
            "or" => b.ir_logical_or(&lf, &rf),
            "mat_mul" => crate::ops::static_ndarray_ops::matmul(b, &lf, &rf),
            _ => panic!("Unknown binary operator: {}", op),
        }
    } else {
        // Integer operations
        match op {
            "add" => b.ir_add_i(lhs, rhs),
            "sub" => b.ir_sub_i(lhs, rhs),
            "mul" => b.ir_mul_i(lhs, rhs),
            "div" => b.ir_div_i(lhs, rhs),
            "mod" => b.ir_mod_i(lhs, rhs),
            "floor_div" => b.ir_floor_div_i(lhs, rhs),
            "pow" => b.ir_pow_i(lhs, rhs),
            "eq" => b.ir_equal_i(lhs, rhs),
            "ne" => b.ir_not_equal_i(lhs, rhs),
            "lt" => b.ir_less_than_i(lhs, rhs),
            "lte" => b.ir_less_than_or_equal_i(lhs, rhs),
            "gt" => b.ir_greater_than_i(lhs, rhs),
            "gte" => b.ir_greater_than_or_equal_i(lhs, rhs),
            "and" => b.ir_logical_and(lhs, rhs),
            "or" => b.ir_logical_or(lhs, rhs),
            "mat_mul" => crate::ops::static_ndarray_ops::matmul(b, lhs, rhs),
            _ => panic!("Unknown binary operator: {}", op),
        }
    }
}

/// Composite comparison (eq, ne, lt, lte, gt, gte) — element-wise then reduce.
pub fn composite_comparison(b: &mut IRBuilder, op: &str, ld: &CompositeData, rd: &CompositeData) -> Value {
    let min_len = ld.values.len().min(rd.values.len());
    match op {
        "eq" => {
            if ld.values.len() != rd.values.len() {
                return b.ir_constant_bool(false);
            }
            let mut result = b.ir_constant_bool(true);
            for i in 0..min_len {
                let cmp = apply_binary_op(b, "eq", &ld.values[i], &rd.values[i]);
                let cmp_bool = to_scalar_bool(b, &cmp);
                result = b.ir_logical_and(&result, &cmp_bool);
            }
            result
        }
        "ne" => {
            let eq = composite_comparison(b, "eq", ld, rd);
            b.ir_logical_not(&eq)
        }
        "lt" | "lte" | "gt" | "gte" => {
            // Lexicographic comparison
            // For simplicity, compare element by element
            if min_len == 0 {
                return match op {
                    "lt" => b.ir_constant_bool(ld.values.len() < rd.values.len()),
                    "lte" => b.ir_constant_bool(ld.values.len() <= rd.values.len()),
                    "gt" => b.ir_constant_bool(ld.values.len() > rd.values.len()),
                    "gte" => b.ir_constant_bool(ld.values.len() >= rd.values.len()),
                    _ => unreachable!(),
                };
            }
            // Compare first elements
            let mut result = apply_binary_op(b, op, &ld.values[0], &rd.values[0]);
            for i in 1..min_len {
                // If previous elements were equal, compare this element
                let prev_eq = apply_binary_op(b, "eq", &ld.values[i-1], &rd.values[i-1]);
                let prev_eq_bool = to_scalar_bool(b, &prev_eq);
                let this_cmp = apply_binary_op(b, op, &ld.values[i], &rd.values[i]);
                let this_cmp_bool = to_scalar_bool(b, &this_cmp);
                // result = prev_eq ? this_cmp : result
                result = b.ir_select_i(&prev_eq_bool, &this_cmp_bool, &result);
            }
            // Handle different lengths: if all common elements are equal
            if ld.values.len() != rd.values.len() {
                let all_eq = composite_comparison(b, "eq",
                    &CompositeData { elements_type: ld.elements_type[..min_len].to_vec(), values: ld.values[..min_len].to_vec() },
                    &CompositeData { elements_type: rd.elements_type[..min_len].to_vec(), values: rd.values[..min_len].to_vec() },
                );
                let all_eq_bool = to_scalar_bool(b, &all_eq);
                let len_result = match op {
                    "lt" | "lte" => b.ir_constant_bool(ld.values.len() < rd.values.len()),
                    "gt" | "gte" => b.ir_constant_bool(ld.values.len() > rd.values.len()),
                    _ => unreachable!(),
                };
                result = b.ir_select_i(&all_eq_bool, &len_result, &result);
            }
            result
        }
        _ => b.ir_constant_bool(false),
    }
}

/// For larger arrays: uses DynamicNDArrayGetItem IR (lowered to memory by opt pass).
pub fn dynamic_list_subscript(b: &mut IRBuilder, data: &CompositeData, idx: &Value) -> Value {
    if data.values.is_empty() {
        return Value::None;
    }
    let n = data.values.len();

    if n < 100 {
        // Mux path: SelectI chain
        let mut result = data.values.last().unwrap().clone();
        for i in (0..n - 1).rev() {
            let const_i = b.ir_constant_int(i as i64);
            let cmp = b.ir_equal_i(idx, &const_i);
            result = b.ir_select_i(&cmp, &data.values[i], &result);
        }
        result
    } else {
        // Memory path: allocate segment, write all values, read at dynamic index
        let seg_id = b.alloc_segment_id();
        let arr_id = b.alloc_array_id();

        // Allocate memory segment
        b.ir_allocate_memory(seg_id, n as u32, 0);

        // Write all values to the segment
        for (i, val) in data.values.iter().enumerate() {
            let addr = b.ir_constant_int(i as i64);
            b.ir_write_memory(seg_id, &addr, val);
        }

        // Read at dynamic index using DynamicNDArrayGetItem
        b.ir_dynamic_ndarray_get_item(arr_id, seg_id, idx)
    }
}

/// Returns the updated composite.
pub fn dynamic_list_set_item(b: &mut IRBuilder, data: &CompositeData, idx: &Value, value: &Value) -> Value {
    let n = data.values.len();
    if n == 0 {
        return Value::List(data.clone());
    }

    if n < 100 {
        // Mux path: for each position, conditionally replace
        let mut new_values = Vec::new();
        let mut new_types = Vec::new();
        for i in 0..n {
            let const_i = b.ir_constant_int(i as i64);
            let cmp = b.ir_equal_i(idx, &const_i);
            let selected = b.ir_select_i(&cmp, value, &data.values[i]);
            new_types.push(selected.zinnia_type());
            new_values.push(selected);
        }
        Value::List(CompositeData { elements_type: new_types, values: new_values })
    } else {
        // Memory path: allocate, write all, then overwrite at dynamic index, read all back
        let seg_id = b.alloc_segment_id();
        let arr_id = b.alloc_array_id();

        // Allocate memory segment
        b.ir_allocate_memory(seg_id, n as u32, 0);

        // Write all original values
        for (i, val) in data.values.iter().enumerate() {
            let addr = b.ir_constant_int(i as i64);
            b.ir_write_memory(seg_id, &addr, val);
        }

        // Write the new value at the dynamic index
        b.ir_dynamic_ndarray_set_item(arr_id, seg_id, idx, value);

        // Read all values back to reconstruct the list
        let mut new_values = Vec::new();
        let mut new_types = Vec::new();
        for i in 0..n {
            let addr = b.ir_constant_int(i as i64);
            let read_val = b.ir_read_memory(seg_id, &addr);
            new_types.push(read_val.zinnia_type());
            new_values.push(read_val);
        }
        Value::List(CompositeData { elements_type: new_types, values: new_values })
    }
}

/// Returns true if `val` is a composite (or scalar) whose every leaf is a
/// numeric value (Integer/Float/Boolean). Used to distinguish "ndarray-like"
/// composites from heterogeneous Python lists, so we can pick NumPy-style
/// broadcasting vs. Python list concatenation in the binary-op dispatcher.
pub fn is_numeric_composite(val: &Value) -> bool {
    match val {
        Value::List(data) | Value::Tuple(data) => {
            data.values.iter().all(is_numeric_composite)
        }
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => true,
        _ => false,
    }
}

/// Ensure a value is a float, casting if necessary.
pub fn ensure_float(b: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::Float(_) => val.clone(),
        _ => b.ir_float_cast(val),
    }
}
