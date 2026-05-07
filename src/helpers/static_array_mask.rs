//! Native boolean-mask read / write paths for `Value::StaticArray`.
//!
//! P5b of `compiler.epic-segment-native-static-arrays`: this module adds a
//! first-class boolean-masking implementation that operates directly on the
//! cached payload of segment-backed StaticArrays, avoiding the legacy
//! `boolean_mask_static` / `dyn_filter` round-trip through `Value::List`.
//!
//! ## Output type contract
//!
//! - **Static mask** (every cached cell carries a compile-time
//!   `static_val` — typically a constructor like `np.array([True, False,
//!   True])` or a comparison whose operands are constant): the surviving
//!   cell count is known at compile time, so the output is a fresh
//!   `Value::StaticArray` with `shape == [true_count]`. Cells of the
//!   surviving positions are placed in the new payload in original order.
//!
//! - **Dynamic mask** (one or more mask cells lack a compile-time
//!   `static_val` — e.g. comparison against a runtime input): the surviving
//!   cell count is data-dependent, so the output is a `Value::DynamicNDArray`
//!   with `total_bound == prod(arr.shape)`. The segment is built via the
//!   same compaction trick used by `dyn_filter` (write-pointer advances
//!   only on truthy mask cells; non-kept positions leave a placeholder
//!   that will either be overwritten by the next kept element or sit
//!   beyond `runtime_length`).
//!
//! ## Boolean-mask write
//!
//! `arr[mask] = v` is a per-cell `select(mask[i], v_i, arr[i])` that writes
//! the result back into `arr.segment_id` at offset `i`. RHS may be a
//! scalar (broadcast across all true positions) or another StaticArray /
//! List with as many leaves as the array has cells (numpy semantics: the
//! RHS for a boolean-mask write is sized to `true_count`, but for the
//! static-shape variant we accept either a same-shape RHS — common from
//! `arr[mask] = arr2[mask]` — or a true-count-sized RHS, broadcasting via
//! a running selection counter).
//!
//! Cache invalidation mirrors P3 — the mask write touches a runtime
//! pattern of cells, so the segment's cache entry is dropped (option (a)
//! from the P3 card).

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value,
};

use super::static_array_elementwise::payload_cells;

// ────────────────────────────────────────────────────────────────────────
// Mask classification
// ────────────────────────────────────────────────────────────────────────

/// Returns `true` when every cell carries a compile-time `static_val`. For a
/// Boolean StaticArray that means `bool_val()` is `Some(_)` for every cached
/// cell. For an Integer-storage Boolean (e.g. P4a comparison output whose
/// cells are still Boolean wires) we accept either `bool_val()` or
/// `int_val()`.
fn mask_is_fully_static(cells: &[Value]) -> bool {
    cells.iter().all(|c| match c {
        Value::Boolean(s) => s.static_val.is_some(),
        Value::Integer(s) => s.static_val.is_some(),
        _ => false,
    })
}

/// Pull the truthy/falsy compile-time value of a cell. Caller must have
/// established that `mask_is_fully_static` returned true.
fn cell_static_truth(c: &Value) -> bool {
    match c {
        Value::Boolean(s) => s.static_val.unwrap(),
        Value::Integer(s) => s.static_val.unwrap() != 0,
        _ => panic!("cell_static_truth: caller must guarantee static cell"),
    }
}

/// Detect whether `mask` is a Boolean-shaped StaticArray. We accept any
/// StaticArray dtype because P4a comparisons output `dtype: Integer` with
/// `Value::Boolean` cells; we determine "boolean-ness" by inspecting the
/// cached cells, not the dtype tag.
pub fn is_boolean_mask_static_array(b: &mut IRBuilder, mask: &Value) -> bool {
    let segment_id = match mask {
        Value::StaticArray { segment_id, .. } => *segment_id,
        _ => return false,
    };
    if let Some(cells) = b.static_array_payload.get(&segment_id) {
        if cells.is_empty() {
            return false;
        }
        return cells.iter().all(|c| matches!(c, Value::Boolean(_)));
    }
    // No cache entry — conservatively say no, the legacy path will pick
    // it up. (In practice the cache is always populated for a StaticArray
    // that came out of `build_static_array_from_flat`.)
    false
}

// ────────────────────────────────────────────────────────────────────────
// Public entry points
// ────────────────────────────────────────────────────────────────────────

/// Try to apply a boolean-mask read on a `Value::StaticArray`.
///
/// Returns `Some(result)` when both `arr` and `mask` are StaticArrays of
/// matching shape AND the mask cells are Boolean wires. The result is
/// either a `Value::StaticArray` (static mask) or a `Value::DynamicNDArray`
/// (dynamic mask).
pub fn try_apply_boolean_mask_read(
    b: &mut IRBuilder,
    arr: &Value,
    mask: &Value,
) -> Option<Value> {
    let (arr_dtype, arr_shape, _arr_seg) = match arr {
        Value::StaticArray { dtype, shape, segment_id, .. } => {
            (*dtype, shape.clone(), *segment_id)
        }
        _ => return None,
    };
    let mask_shape = match mask {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => return None,
    };
    if !is_boolean_mask_static_array(b, mask) {
        return None;
    }
    if arr_shape != mask_shape {
        // numpy raises an IndexError. Surface a clear panic; downstream
        // callers will see this as a hard compile error.
        panic!(
            "boolean index did not match indexed array along dimensions; \
             array shape {:?}, mask shape {:?}",
            arr_shape, mask_shape
        );
    }
    // Complex mask read is rejected for now (there's no defined Complex
    // ordering / mask interaction in numpy beyond eq / ne, which already
    // produces a Boolean mask via the comparison path).
    if matches!(arr_dtype, NumberType::Complex) {
        return None;
    }

    let arr_cells = payload_cells(b, arr);
    let mask_cells = payload_cells(b, mask);
    let total: usize = arr_shape.iter().product();
    debug_assert_eq!(arr_cells.len(), total);
    debug_assert_eq!(mask_cells.len(), total);

    if mask_is_fully_static(&mask_cells) {
        // Static fast path: surviving cell count known at compile time.
        let mut surviving: Vec<Value> = Vec::new();
        for i in 0..total {
            if cell_static_truth(&mask_cells[i]) {
                surviving.push(arr_cells[i].clone());
            }
        }
        let out_shape = vec![surviving.len()];
        return Some(super::static_array::build_static_array_from_flat(
            b, surviving, out_shape, arr_dtype,
        ));
    }

    // Dynamic mask: produce a DynamicNDArray with total_bound == arr.size.
    Some(dynamic_mask_read(b, &arr_cells, &mask_cells, arr_dtype, total))
}

/// Try to apply a boolean-mask write on a `Value::StaticArray`.
///
/// Returns `Some(arr)` (the same StaticArray, with its segment mutated) when
/// `arr` is a StaticArray and `mask` is a matching-shape Boolean mask. RHS
/// may be a scalar (broadcast) or a StaticArray / List of cells.
///
/// Cache invalidation: the segment cache is dropped because the post-write
/// pattern is data-dependent (any cell could have been overwritten if
/// the mask is dynamic; even for a static mask the write changes a
/// compile-time-known subset of cells).
pub fn try_apply_boolean_mask_write(
    b: &mut IRBuilder,
    arr: &Value,
    mask: &Value,
    value: &Value,
) -> Option<Value> {
    let (arr_dtype, arr_shape, arr_seg, _arr_strides, arr_offset, imag_seg) = match arr {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => return None,
    };
    let mask_shape = match mask {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => return None,
    };
    if !is_boolean_mask_static_array(b, mask) {
        return None;
    }
    if arr_shape != mask_shape {
        panic!(
            "boolean index did not match indexed array along dimensions; \
             array shape {:?}, mask shape {:?}",
            arr_shape, mask_shape
        );
    }

    let arr_cells = payload_cells(b, arr);
    let mask_cells = payload_cells(b, mask);
    let total: usize = arr_shape.iter().product();

    // Resolve RHS into a flat Vec<Value>. Two shapes are accepted:
    //   - scalar (one element, broadcast across all true positions)
    //   - same-shape array (one cell per arr cell — common in
    //     `arr[mask] = arr2[mask]` after the read fast-path, since the
    //     read may return a same-true-count array which we need to
    //     conditionally place back into arr).
    enum Rhs<'a> {
        Scalar(&'a Value),
        FullShape(Vec<Value>),
        TrueCount(Vec<Value>),
    }
    let rhs = if value.is_number() {
        Rhs::Scalar(value)
    } else if let Value::StaticArray { shape: rs, .. } = value {
        let cells = payload_cells(b, value);
        if rs == &arr_shape {
            Rhs::FullShape(cells)
        } else {
            Rhs::TrueCount(cells)
        }
    } else if matches!(value, Value::List(_) | Value::Tuple(_)) {
        let leaves = super::composite::flatten_composite(value);
        if leaves.len() == total {
            Rhs::FullShape(leaves)
        } else {
            Rhs::TrueCount(leaves)
        }
    } else {
        // Unknown RHS shape — punt to legacy path.
        return None;
    };

    // Static fast-path bookkeeping for the TrueCount RHS form: walk the
    // mask's static cells to align rhs cells to the right positions.
    let static_mask = mask_is_fully_static(&mask_cells);

    if matches!(arr_dtype, NumberType::Complex) {
        // Component-aware write across both segments.
        let imag_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
        let mut rhs_idx_static: usize = 0;
        for i in 0..total {
            let abs = arr_offset + i;
            let addr = b.ir_constant_int(abs as i64);
            let mask_val = &mask_cells[i];
            let mask_bool = ensure_bool(b, mask_val);
            // Determine the per-cell new value.
            let new_val: Value = match &rhs {
                Rhs::Scalar(v) => (*v).clone(),
                Rhs::FullShape(cells) => cells[i].clone(),
                Rhs::TrueCount(cells) => {
                    // For static mask, pick by surviving count. For dynamic
                    // mask the static-count path doesn't apply — fall
                    // through to legacy if we got a TrueCount RHS with a
                    // dynamic mask.
                    if !static_mask {
                        return None;
                    }
                    let pick = if cell_static_truth(mask_val) {
                        let v = cells.get(rhs_idx_static).cloned();
                        rhs_idx_static += 1;
                        v
                    } else {
                        None
                    };
                    pick.unwrap_or_else(|| arr_cells[i].clone())
                }
            };
            let (new_re, new_im) =
                crate::helpers::value_ops::unpack_value_to_complex_parts(b, &new_val);
            let (cur_re, cur_im) =
                crate::helpers::value_ops::unpack_value_to_complex_parts(b, &arr_cells[i]);
            let sel_re = b.ir_select_f(&mask_bool, &new_re, &cur_re);
            let sel_im = b.ir_select_f(&mask_bool, &new_im, &cur_im);
            b.ir_write_memory(arr_seg, &addr, &sel_re);
            b.ir_write_memory(imag_seg, &addr, &sel_im);
        }
        b.static_array_payload.remove(&arr_seg);
        return Some(arr.clone());
    }

    // Non-Complex (Integer / Float) per-cell select + write.
    let mut rhs_idx_static: usize = 0;
    for i in 0..total {
        let abs = arr_offset + i;
        let addr = b.ir_constant_int(abs as i64);
        let mask_val = &mask_cells[i];
        let mask_bool = ensure_bool(b, mask_val);
        let new_raw: Value = match &rhs {
            Rhs::Scalar(v) => (*v).clone(),
            Rhs::FullShape(cells) => cells[i].clone(),
            Rhs::TrueCount(cells) => {
                if !static_mask {
                    return None;
                }
                let pick = if cell_static_truth(mask_val) {
                    let v = cells.get(rhs_idx_static).cloned();
                    rhs_idx_static += 1;
                    v
                } else {
                    None
                };
                pick.unwrap_or_else(|| arr_cells[i].clone())
            }
        };
        let new_cast = cast_to_dtype(b, &new_raw, arr_dtype);
        let cur_cast = cast_to_dtype(b, &arr_cells[i], arr_dtype);
        let selected = if matches!(arr_dtype, NumberType::Float) {
            b.ir_select_f(&mask_bool, &new_cast, &cur_cast)
        } else {
            b.ir_select_i(&mask_bool, &new_cast, &cur_cast)
        };
        b.ir_write_memory(arr_seg, &addr, &selected);
    }
    b.static_array_payload.remove(&arr_seg);
    Some(arr.clone())
}

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn ensure_bool(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Boolean(_) => v.clone(),
        _ => b.ir_bool_cast(v),
    }
}

fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) { v.clone() } else { b.ir_float_cast(v) }
        }
        NumberType::Integer => {
            if matches!(v, Value::Integer(_)) {
                v.clone()
            } else if matches!(v, Value::Boolean(_)) {
                let casted = b.ir_int_cast(v);
                if let (Value::Integer(mut s), Some(bv)) = (casted.clone(), v.bool_val()) {
                    s.static_val = Some(if bv { 1 } else { 0 });
                    Value::Integer(s)
                } else {
                    casted
                }
            } else {
                b.ir_int_cast(v)
            }
        }
        NumberType::Complex => unreachable!("cast_to_dtype(Complex) used outside Complex path"),
    }
}

/// Build a DynamicNDArray output from a per-cell mask read where the mask is
/// not fully compile-time-known. Mirrors `dyn_filter`'s compaction trick.
fn dynamic_mask_read(
    b: &mut IRBuilder,
    arr_cells: &[Value],
    mask_cells: &[Value],
    dtype: NumberType,
    total: usize,
) -> Value {
    let default_val = match dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Complex => unreachable!("Complex mask read rejected upstream"),
    };
    let default_sv = value_to_scalar_i64(&default_val);
    let default_elements = vec![default_sv; total];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &default_elements, dtype);

    let mut write_ptr = b.ir_constant_int(0);
    let one = b.ir_constant_int(1);
    let zero = b.ir_constant_int(0);
    for i in 0..total {
        let mask_bool = ensure_bool(b, &mask_cells[i]);
        let cell_cast = cast_to_dtype(b, &arr_cells[i], dtype);
        let val_to_write = if dtype == NumberType::Float {
            b.ir_select_f(&mask_bool, &cell_cast, &default_val)
        } else {
            b.ir_select_i(&mask_bool, &cell_cast, &default_val)
        };
        b.ir_write_memory(segment_id, &write_ptr, &val_to_write);
        let inc = b.ir_select_i(&mask_bool, &one, &zero);
        write_ptr = b.ir_add_i(&write_ptr, &inc);
    }

    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(&mut b.dim_table, 0, total)],
        total,
    );
    let runtime_len = value_to_scalar_i64(&write_ptr);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![total],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: runtime_len.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_len],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositeData;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,
        })
    }

    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let lst = list_of(leaves);
        crate::helpers::static_array::to_static_array(b, &lst).unwrap()
    }

    fn make_1d_bool(b: &mut IRBuilder, bits: &[bool]) -> Value {
        let leaves: Vec<Value> = bits.iter().map(|x| b.ir_constant_bool(*x)).collect();
        let lst = list_of(leaves);
        crate::helpers::static_array::to_static_array(b, &lst).unwrap()
    }

    #[test]
    fn boolean_construction_keeps_boolean_cells() {
        let mut b = IRBuilder::new();
        let m = make_1d_bool(&mut b, &[true, false, true]);
        match &m {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &m);
        for c in &cells {
            assert!(matches!(c, Value::Boolean(_)));
        }
        assert!(is_boolean_mask_static_array(&mut b, &m));
    }

    #[test]
    fn static_mask_read_returns_static_array() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[10, 20, 30, 40]);
        let mask = make_1d_bool(&mut b, &[true, false, true, false]);
        let out = try_apply_boolean_mask_read(&mut b, &arr, &mask).expect("native path");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2]),
            _ => panic!("expected static StaticArray output"),
        }
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(10), Some(30)]);
    }

    #[test]
    fn static_mask_write_scalar() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[10, 20, 30, 40]);
        let mask = make_1d_bool(&mut b, &[true, false, true, false]);
        let zero = b.ir_constant_int(0);
        let out = try_apply_boolean_mask_write(&mut b, &arr, &mask, &zero).expect("native path");
        // Cache invalidated; next read goes through ir_read_memory. We
        // don't have an easy way to read post-write here without running
        // the proving backend, so just confirm we got back a StaticArray.
        match out {
            Value::StaticArray { .. } => {}
            _ => panic!("expected StaticArray return"),
        }
    }

    #[test]
    fn returns_none_for_non_boolean_mask() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[1, 2, 3]);
        let other_int = make_1d_int(&mut b, &[0, 1, 1]);
        assert!(try_apply_boolean_mask_read(&mut b, &arr, &other_int).is_none());
    }
}
