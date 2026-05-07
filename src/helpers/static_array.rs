//! Conversion helpers between `Value::List` (numeric leaves) and the
//! segment-backed `Value::StaticArray`.
//!
//! These are the boundary shims for P1 of the
//! `compiler.epic-segment-native-static-arrays` epic. Constructors and the
//! input-parser path emit `Value::StaticArray`; legacy ops that still
//! consume nested `Value::List` use [`to_value_list`] at their entry point
//! to fall back onto the materialised list view. Both directions preserve
//! per-leaf `ScalarValue::static_val` constant-fold information.
//!
//! Once P6 lands and the legacy `Vec<Value>` numeric path is deleted,
//! `to_value_list` can go away — but [`to_static_array`] stays as the
//! constructor entry point.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::{scalar_i64_to_value, value_to_scalar_i64};
use crate::types::{CompositeData, NumberType, ScalarValue, Value, ZinniaType};

use super::composite::{flatten_composite, get_composite_shape};
use super::segment::alloc_and_write;
use super::shape_arith::row_major_strides;

/// Determine whether a value is a (possibly-nested) composite of purely
/// numeric leaves (Integer / Float / Boolean / Complex — P5a admits Complex).
/// Heterogeneous lists, lists containing strings, lists of arrays, etc. all
/// return `false`.
fn is_pure_numeric_composite(val: &Value) -> bool {
    match val {
        Value::List(d) | Value::Tuple(d) => {
            !d.values.is_empty() && d.values.iter().all(is_pure_numeric_composite)
        }
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::Complex { .. } => true,
        _ => false,
    }
}

/// Returns true if any leaf in the composite is `Value::Complex`.
fn composite_has_complex(val: &Value) -> bool {
    match val {
        Value::Complex { .. } => true,
        Value::List(d) | Value::Tuple(d) => d.values.iter().any(composite_has_complex),
        _ => false,
    }
}

/// Infer the dtype for a numeric composite. Any `Float` leaf forces `Float`;
/// otherwise `Integer` (Boolean folds into Integer storage).
fn infer_numeric_dtype(val: &Value) -> NumberType {
    let leaves = flatten_composite(val);
    if leaves.iter().any(|v| matches!(v, Value::Complex { .. })) {
        NumberType::Complex
    } else if leaves.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

/// Build a `Value::StaticArray` for a precomputed flat numeric payload and
/// shape. Used by constructors that already have the raw cells.
///
/// The flat payload `Value`s are also recorded in the builder's
/// `static_array_payload` cache so the boundary shim
/// ([`to_value_list`]) can return the original wires without issuing N
/// `ir_read_memory` ops per call. The segment is still written so dynamic
/// indexing (P2/P3) and the proving backend see the populated zkRAM cells.
///
/// Constant-fill optimisation: if every cell carries the same compile-time
/// value, the segment is allocated with that value as the `init_value` and
/// the per-cell `ir_write_memory` ops are skipped. Saves O(N) IR for
/// `np.zeros((n, n))`-style constructors on large shapes.
pub fn build_static_array_from_flat(
    b: &mut IRBuilder,
    flat: Vec<Value>,
    shape: Vec<usize>,
    dtype: NumberType,
) -> Value {
    let cells: Vec<ScalarValue<i64>> = flat.iter().map(value_to_scalar_i64).collect();
    let segment_id = build_segment_for_payload(b, &cells, dtype);
    let strides = row_major_strides(&shape);
    b.static_array_payload.insert(segment_id, flat);
    Value::StaticArray {
        dtype,
        shape,
        segment_id,
        strides,
        offset: 0,
        imag_segment_id: None,
    }
}

/// Build a Complex `Value::StaticArray` from precomputed flat real and imag
/// payloads (same length) and shape. Allocates two parallel segments and
/// caches the original Complex wires under the *real* segment id (cache
/// representation A — see P5a card / segarr-complex-dtype README).
pub fn build_static_array_from_flat_complex(
    b: &mut IRBuilder,
    real_flat: Vec<Value>,
    imag_flat: Vec<Value>,
    shape: Vec<usize>,
) -> Value {
    assert_eq!(real_flat.len(), imag_flat.len());
    let real_cells: Vec<ScalarValue<i64>> = real_flat.iter().map(value_to_scalar_i64).collect();
    let imag_cells: Vec<ScalarValue<i64>> = imag_flat.iter().map(value_to_scalar_i64).collect();
    let real_seg = build_segment_for_payload(b, &real_cells, NumberType::Float);
    let imag_seg = build_segment_for_payload(b, &imag_cells, NumberType::Float);
    let strides = row_major_strides(&shape);

    // Cache representation (A): one entry keyed on the *real* segment_id,
    // holding `Vec<Value::Complex>` so payload_cells / to_value_list lookups
    // see the original Complex scalars.
    let complex_cells: Vec<Value> = real_flat
        .iter()
        .zip(imag_flat.iter())
        .map(|(re, im)| {
            let r_sv = match re {
                Value::Float(s) => s.clone(),
                _ => {
                    let f = b.ir_float_cast(re);
                    match f { Value::Float(s) => s, _ => unreachable!() }
                }
            };
            let i_sv = match im {
                Value::Float(s) => s.clone(),
                _ => {
                    let f = b.ir_float_cast(im);
                    match f { Value::Float(s) => s, _ => unreachable!() }
                }
            };
            Value::Complex { real: r_sv, imag: i_sv }
        })
        .collect();
    b.static_array_payload.insert(real_seg, complex_cells);

    Value::StaticArray {
        dtype: NumberType::Complex,
        shape,
        segment_id: real_seg,
        strides,
        offset: 0,
        imag_segment_id: Some(imag_seg),
    }
}

/// Allocate a segment for a fixed payload, taking advantage of
/// `ir_allocate_memory`'s `init_value` to skip per-cell writes when every
/// cell shares the same compile-time constant. Falls back to
/// `alloc_and_write` for non-uniform payloads.
fn build_segment_for_payload(
    b: &mut IRBuilder,
    cells: &[ScalarValue<i64>],
    dtype: NumberType,
) -> u32 {
    // All-equal-known-constant fast path. Encoding nuance: `init_value` is
    // an `i64`. For Float arrays we re-interpret the static_val of each
    // cell back to its i64 representation; if the payload is non-uniform
    // we fall through.
    if !cells.is_empty() {
        if let Some(first_val) = cells[0].static_val {
            let all_match = cells.iter().all(|c| c.static_val == Some(first_val));
            if all_match {
                let seg = b.alloc_segment_id();
                b.ir_allocate_memory(seg, cells.len() as u32, first_val);
                return seg;
            }
        }
    }
    alloc_and_write(b, cells, dtype)
}

/// Convert a `Value::List` / `Value::Tuple` of numeric leaves into a
/// `Value::StaticArray`. Returns `None` if the value isn't a pure-numeric
/// static composite (heterogeneous types, contains strings, etc.). Pass
/// through if already a `Value::StaticArray`.
///
/// Constant-fold information is preserved per leaf via
/// [`value_to_scalar_i64`].
pub fn to_static_array(b: &mut IRBuilder, val: &Value) -> Option<Value> {
    if let Value::StaticArray { .. } = val {
        return Some(val.clone());
    }
    if !is_pure_numeric_composite(val) {
        return None;
    }
    let shape = get_composite_shape(val);
    let dtype = infer_numeric_dtype(val);
    let flat = flatten_composite(val);
    if dtype == NumberType::Complex {
        // Promote each leaf to Value::Complex, then split into reals / imags.
        let mut reals = Vec::with_capacity(flat.len());
        let mut imags = Vec::with_capacity(flat.len());
        for leaf in &flat {
            match leaf {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real.clone()));
                    imags.push(Value::Float(imag.clone()));
                }
                Value::Float(s) => {
                    reals.push(Value::Float(s.clone()));
                    imags.push(b.ir_constant_float(0.0));
                }
                Value::Integer(_) | Value::Boolean(_) => {
                    let r = b.ir_float_cast(leaf);
                    reals.push(r);
                    imags.push(b.ir_constant_float(0.0));
                }
                _ => unreachable!("is_pure_numeric_composite already guarded leaf types"),
            }
        }
        return Some(build_static_array_from_flat_complex(b, reals, imags, shape));
    }
    Some(build_static_array_from_flat(b, flat, shape, dtype))
}

/// Recursively convert any `Value::StaticArray` (top-level or nested inside
/// a `List` / `Tuple`) into a nested `Value::List`. Useful when a legacy op
/// receives a List of arrays and any of its inner elements may be a
/// segment-backed StaticArray. Non-StaticArray values pass through.
pub fn deep_to_value_list(b: &mut IRBuilder, val: &Value) -> Value {
    // P6 fast path: skip the recursive walk and return the value untouched
    // when there's no segment-backed array hiding inside. The common case
    // for the boundary shim is "no StaticArray here, do nothing".
    if !contains_static_array(val) {
        return val.clone();
    }
    match val {
        Value::StaticArray { .. } => to_value_list(b, val),
        Value::List(data) => {
            let new_vals: Vec<Value> = data.values.iter().map(|v| deep_to_value_list(b, v)).collect();
            let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: new_types, values: new_vals })
        }
        Value::Tuple(data) => {
            let new_vals: Vec<Value> = data.values.iter().map(|v| deep_to_value_list(b, v)).collect();
            let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::Tuple(CompositeData { elements_type: new_types, values: new_vals })
        }
        _ => val.clone(),
    }
}

/// Returns true if `val` is a `StaticArray` or contains one anywhere in a
/// nested `List` / `Tuple`. P6: used to avoid the deep clone in
/// [`deep_to_value_list`] when there's nothing segment-backed to materialise.
fn contains_static_array(val: &Value) -> bool {
    match val {
        Value::StaticArray { .. } => true,
        Value::List(data) | Value::Tuple(data) => {
            data.values.iter().any(contains_static_array)
        }
        _ => false,
    }
}

/// Read the segment payload of a `Value::StaticArray` and rebuild a nested
/// `Value::List` that matches the legacy numeric-array representation.
/// No-op for non-`StaticArray` values (returned as-is).
///
/// Lossy on segment IR pointers: each leaf is materialised as a fresh
/// `ScalarValue` (`static_val` carries through, `ptr` is the segment-read
/// statement). Legacy ops reading the resulting list see the same compile-
/// time constants they would have seen if the array had been built as a
/// `Value::List` to begin with.
pub fn to_value_list(b: &mut IRBuilder, val: &Value) -> Value {
    let (dtype, shape, segment_id, _strides, offset, imag_seg) = match val {
        Value::StaticArray {
            dtype,
            shape,
            segment_id,
            strides,
            offset,
            imag_segment_id,
        } => (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id),
        _ => return val.clone(),
    };
    let total: usize = shape.iter().product();
    // P1 fast-path: if the segment payload is cached (always true for arrays
    // built via `build_static_array_from_flat`), reuse the original wires
    // instead of issuing N segment reads. This keeps the boundary cheap so
    // benchmarks with heavy indexing don't pay quadratic IR growth.
    let leaves: Vec<Value> = if let Some(cached) = b.static_array_payload.get(&segment_id) {
        cached
            .iter()
            .skip(offset)
            .take(total)
            .cloned()
            .collect()
    } else if dtype == NumberType::Complex {
        // P5a: dual-segment fallback for Complex when cache was invalidated.
        let im_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
        let mut tmp = Vec::with_capacity(total);
        for i in 0..total {
            tmp.push(super::static_array_read::read_complex_leaf(b, segment_id, im_seg, offset + i));
        }
        tmp
    } else {
        // Fallback: materialise via segment reads (only relevant for
        // StaticArrays created without cache registration).
        let mut tmp = Vec::with_capacity(total);
        for i in 0..total {
            let addr = b.ir_constant_int((offset + i) as i64);
            let raw = b.ir_read_memory(segment_id, &addr);
            let sv = value_to_scalar_i64(&raw);
            let leaf = scalar_i64_to_value(&sv, dtype);
            tmp.push(leaf);
        }
        tmp
    };
    let leaf_types: Vec<ZinniaType> = leaves.iter().map(|v| v.zinnia_type()).collect();
    if shape.len() <= 1 {
        return Value::List(CompositeData {
            elements_type: leaf_types,
            values: leaves,
        });
    }
    // Build nested List from flat payload.
    super::composite::build_nested_value(leaves, leaf_types, &shape)
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositeData;

    fn int_value(b: &mut IRBuilder, n: i64) -> Value {
        b.ir_constant_int(n)
    }

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,
        })
    }

    #[test]
    fn round_trip_1d_integer() {
        let mut b = IRBuilder::new();
        let lst = list_of((0..4).map(|i| int_value(&mut b, i)).collect());
        let sa = to_static_array(&mut b, &lst).expect("expected StaticArray");
        match &sa {
            Value::StaticArray { dtype, shape, .. } => {
                assert_eq!(*dtype, NumberType::Integer);
                assert_eq!(*shape, vec![4]);
            }
            _ => panic!("expected StaticArray"),
        }
        let back = to_value_list(&mut b, &sa);
        if let Value::List(data) = back {
            assert_eq!(data.values.len(), 4);
            // With the payload cache, the leaves are the *original* wires,
            // so static_val does survive the round-trip when the source
            // value was constant.
            for (i, v) in data.values.iter().enumerate() {
                assert!(matches!(v, Value::Integer(_)));
                assert_eq!(v.int_val(), Some(i as i64));
            }
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn round_trip_2d_integer() {
        let mut b = IRBuilder::new();
        let row0 = list_of((0..3).map(|i| int_value(&mut b, i)).collect());
        let row1 = list_of((10..13).map(|i| int_value(&mut b, i)).collect());
        let mat = list_of(vec![row0, row1]);
        let sa = to_static_array(&mut b, &mat).expect("expected StaticArray");
        match &sa {
            Value::StaticArray { shape, strides, .. } => {
                assert_eq!(*shape, vec![2, 3]);
                assert_eq!(*strides, vec![3, 1]);
            }
            _ => panic!("expected StaticArray"),
        }
        let back = to_value_list(&mut b, &sa);
        if let Value::List(data) = back {
            assert_eq!(data.values.len(), 2);
            if let Value::List(r0) = &data.values[0] {
                assert_eq!(r0.values.len(), 3);
                assert_eq!(r0.values[0].int_val(), Some(0));
                assert_eq!(r0.values[2].int_val(), Some(2));
            } else {
                panic!("expected nested List row");
            }
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn pass_through_static_array() {
        let mut b = IRBuilder::new();
        let lst = list_of((0..3).map(|i| int_value(&mut b, i)).collect());
        let sa = to_static_array(&mut b, &lst).unwrap();
        let sa2 = to_static_array(&mut b, &sa).unwrap();
        match (&sa, &sa2) {
            (
                Value::StaticArray { segment_id: a, .. },
                Value::StaticArray { segment_id: b, .. },
            ) => assert_eq!(a, b),
            _ => panic!(),
        }
    }

    #[test]
    fn rejects_heterogeneous() {
        let mut b = IRBuilder::new();
        let s = b.ir_constant_int(1);
        let f = b.ir_constant_float(2.0);
        let mixed = list_of(vec![s, f]);
        // Mixed Integer + Float is still numeric — so this must succeed,
        // promoted to Float.
        let sa = to_static_array(&mut b, &mixed).expect("mixed numeric ok");
        match sa {
            Value::StaticArray { dtype, .. } => assert_eq!(dtype, NumberType::Float),
            _ => panic!(),
        }
    }
}
