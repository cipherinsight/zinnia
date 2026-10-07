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

use crate::builder::{IRBuilder, StaticArrayProductProvenance};
use crate::types::{CompositeData, NumberType, Value, ValueId, ZinniaType};

use super::super::composite::{flatten_composite, get_composite_shape};
use super::super::shape_arith::row_major_strides;

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
/// shape: a new array storage holding the cells (`builder/store.rs`; no IR).
pub fn build_static_array_from_flat(
    b: &mut IRBuilder,
    flat: Vec<Value>,
    shape: Vec<usize>,
    dtype: NumberType,
) -> Value {
    let segment_id = b.alloc_storage(flat.clone(), dtype);
    let strides = row_major_strides(&shape);
    b.note_static_array_cells(segment_id, &flat);
    b.note_segment_cells(segment_id, &flat);
    Value::StaticArray {
        dtype,
        shape,
        segment_id,
        strides,
        offset: 0,
        imag_segment_id: None,
        value_id: ValueId::next(),
    }
}

/// A static array whose cells are the deferred product `provenance`: a
/// storage whose cells are computed when first accessed (a fused consumer
/// may read the factors instead).
pub fn build_static_array_product_provenance(
    b: &mut IRBuilder,
    provenance: StaticArrayProductProvenance,
) -> Value {
    let total: usize = provenance.out_shape.iter().product();
    let zero = b.ir_constant_int(0);
    let segment_id = b.alloc_filled(total, &zero, provenance.dtype);
    let strides = row_major_strides(&provenance.out_shape);
    let dtype = provenance.dtype;
    let shape = provenance.out_shape.clone();
    b.static_array_product_provenance
        .insert(segment_id, provenance);
    Value::StaticArray {
        dtype,
        shape,
        segment_id,
        strides,
        offset: 0,
        imag_segment_id: None,
        value_id: ValueId::next(),
    }
}

/// Build a Complex `Value::StaticArray` from precomputed flat real and imag
/// payloads (same length) and shape: two paired float storages (the known
/// cells of the real one are the complex values).
pub fn build_static_array_from_flat_complex(
    b: &mut IRBuilder,
    real_flat: Vec<Value>,
    imag_flat: Vec<Value>,
    shape: Vec<usize>,
) -> Value {
    assert_eq!(real_flat.len(), imag_flat.len());
    let real_seg = b.alloc_storage(real_flat, NumberType::Float);
    let imag_seg = b.alloc_storage(imag_flat, NumberType::Float);
    b.pair_complex_storages(real_seg, imag_seg);
    let strides = row_major_strides(&shape);
    Value::StaticArray {
        dtype: NumberType::Complex,
        shape,
        segment_id: real_seg,
        strides,
        offset: 0,
        imag_segment_id: Some(imag_seg),
        value_id: ValueId::next(),
    }
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
        // A nested list cannot represent trailing dimensions after a zero
        // axis. Preserve the array metadata for native NumPy dispatch.
        Value::StaticArray { shape, .. } if shape.contains(&0) => val.clone(),
        Value::StaticArray { .. } => to_value_list(b, val),
        Value::List(data) => {
            let new_vals: Vec<Value> = data
                .values
                .iter()
                .map(|v| deep_to_value_list(b, v))
                .collect();
            let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: new_types,
                values: new_vals,
                value_id: ValueId::next(),
            })
        }
        Value::Tuple(data) => {
            let new_vals: Vec<Value> = data
                .values
                .iter()
                .map(|v| deep_to_value_list(b, v))
                .collect();
            let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::Tuple(CompositeData {
                elements_type: new_types,
                values: new_vals,
                value_id: ValueId::next(),
            })
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
        Value::List(data) | Value::Tuple(data) => data.values.iter().any(contains_static_array),
        _ => false,
    }
}

/// The element at the absolute address `abs` of a static array's storage
/// (`segment_id`; for a complex array its real-part storage, paired with the
/// imaginary one): the known cell (no IR), or a read of the storage's
/// current version.
pub(crate) fn element_at(b: &mut IRBuilder, segment_id: u32, abs: usize, dtype: NumberType) -> Value {
    if let Some(cell) = b.known_cell(segment_id, abs) {
        return cell;
    }
    match b.imag_storage(segment_id) {
        Some(imag) => {
            let real = b.storage_read(segment_id, abs);
            let imag = b.storage_read(imag, abs);
            let real = crate::ops::dyn_ndarray::retag(&real, NumberType::Float);
            let imag = crate::ops::dyn_ndarray::retag(&imag, NumberType::Float);
            crate::helpers::value_ops::pack_complex_value(real, imag)
        }
        None => {
            let cell = b.storage_read(segment_id, abs);
            crate::ops::dyn_ndarray::retag(&cell, dtype)
        }
    }
}

/// The elements of a `Value::StaticArray` in logical row-major order as a
/// nested `Value::List` (the legacy numeric-array representation): the
/// known cells, or reads of the storage. No-op for other values.
pub fn to_value_list(b: &mut IRBuilder, val: &Value) -> Value {
    let Value::StaticArray { shape, .. } = val else { return val.clone() };
    let shape = shape.clone();
    let leaves = crate::helpers::static_array::elementwise::payload_cells(b, val);
    let leaf_types: Vec<ZinniaType> = leaves.iter().map(|v| v.zinnia_type()).collect();
    if shape.len() <= 1 {
        return Value::List(CompositeData {
            elements_type: leaf_types,
            values: leaves,

            value_id: ValueId::next(),
        });
    }
    // Build nested List from flat payload.
    super::super::composite::build_nested_value(leaves, leaf_types, &shape)
}

// ────────────────────────────────────────────────────────────────────────
// Shared dynamic-index helpers (read + write)
// ────────────────────────────────────────────────────────────────────────

/// Numpy-wrap a runtime index to the `-axis_len <= i < axis_len` contract:
/// a negative index becomes `i + axis_len`. A literal or provably-non-negative
/// index (per `ask_bounds`) passes through unchanged so we don't emit a
/// needless `select` wire.
///
/// This is the single canonical wrap of the static array paths (reads and
/// writes, through [`checked_axis_index`]).
pub(crate) fn normalize_dynamic_index(b: &mut IRBuilder, idx: &Value, axis_len: usize) -> Value {
    if idx.int_val().is_some() {
        return idx.clone();
    }
    if b.ask_bounds(idx).map(|(min, _)| min >= 0).unwrap_or(false) {
        return idx.clone();
    }
    // An index whose range the facts bound to a few bits has a cheap sign
    // test (the comparison is sized by the range); any other one pins a
    // hinted sign instead of comparing at the full 64-bit width.
    const NARROW: i64 = 1 << 16;
    if b.ask_bounds(idx).is_some_and(|(min, max)| -NARROW < min && max < NARROW) {
        let zero_v = b.ir_constant_int(0);
        let len_v = b.ir_constant_int(axis_len as i64);
        let is_neg = b.ir_less_than_i(idx, &zero_v);
        let wrapped = b.ir_add_i(idx, &len_v);
        return b.ir_select_i(&is_neg, &wrapped, idx);
    }
    wrap_by_pinned_sign(b, idx, axis_len)
}

/// NumPy's position `w = i + n * s` of a runtime index `i` on an axis of
/// length `n`, with the sign `s = [i < 0]` a prover hint (`HintKind::Negative`)
/// instead of a comparison at the full 64-bit width: `s` is asserted to be a
/// bit and `0 <= w < n` is asserted where the code runs (an index outside
/// `[-n, n)` raises in NumPy).
///
/// Soundness: for `i` in `[0, n)` only `s = 0` passes (`i + n >= n`); for `i`
/// in `[-n, 0)` only `s = 1` (with `s = 0`, `w = i` is a field element above
/// `p - n`, outside the unsigned bound); for any other `i` neither does, so
/// there is no witness. So `s` is pinned and `w` is NumPy's position. Cost: a
/// bit, one select and the bound check the access needs anyway.
///
/// On code that runs on every path the position is a placeholder of the
/// memory family (`lower::memory::wrap`), which lowers it this way or by the
/// pinned position one-hot, per target; the check stays here.
fn wrap_by_pinned_sign(b: &mut IRBuilder, idx: &Value, axis_len: usize) -> Value {
    if matches!(b.execution(), crate::builder::guard::Execution::Always) {
        let wrapped = crate::lower::memory::wrap::wrap_lane(b, idx, axis_len);
        let len = b.ir_constant_int(axis_len as i64);
        b.ir_assert_u128_lt(&wrapped, &len);
        return wrapped;
    }
    let zero = b.ir_constant_int(0);
    let negative = b.ir_hint(crate::ir_defs::HintKind::Negative, std::slice::from_ref(idx), &zero);
    b.ir_assert_bit(&negative);
    let len = b.ir_constant_int(axis_len as i64);
    // A select on the bit (one gate, as `i + n s`) keeps `w` one value for
    // the address analyses: a memory read whose address is affine in `w`
    // builds its one-hot over `w`'s `n` values, not over `i` and `s`.
    let shifted = b.ir_add_i(idx, &len);
    let wrapped = b.ir_select_i(&negative, &shifted, idx);
    b.ir_assert_u128_lt(&wrapped, &len);
    wrapped
}

/// Bounds-check an index on axis `axis` (of length `axis_len`) and return
/// its NumPy-wrapped value, the index the access uses.
///
/// NumPy accepts `-axis_len <= idx < axis_len` (a negative index counts
/// from the end) and raises `IndexError` otherwise. This returns
/// `w = idx + axis_len if idx < 0 else idx` and requires `0 <= w <
/// axis_len`, which holds exactly when `idx` is valid: for `idx >= 0` it
/// says `idx < axis_len`, for `idx < 0` it says `-axis_len <= idx`. A
/// runtime index gets the requirement as a resolver discharge (an
/// in-circuit range check unless the facts prove it); a literal index is
/// checked now and rejected with a compile error. Both hold only where the
/// access runs (see `discharge_index_in_range`): under a runtime execution
/// predicate an out-of-range literal makes the circuit assert that the
/// code does not run, and the returned index stays in range where it does
/// not run.
///
/// One index on axes of one length under one execution predicate has one
/// checked position (the first access builds it and its check; later ones
/// reuse it), so accesses through it name the same cells.
///
/// Every index axis must be checked on its own. A check on the flat
/// address alone accepts `a[0, 5]` on a 4x4 array (flat address 5 is in
/// range) and reads `a[1, 1]`; checking each axis makes every address
/// built from the wrapped indices land inside the array (or view).
pub(crate) fn checked_axis_index(
    b: &mut IRBuilder,
    idx: &Value,
    axis: usize,
    axis_len: usize,
    op_name: &'static str,
) -> Value {
    let len = axis_len as i64;
    if let Some(i) = idx.int_val() {
        if !(-len <= i && i < len) {
            if len > 0 && b.assert_never_runs() {
                return b.ir_constant_int(0);
            }
            panic!("index {i} is out of bounds for axis {axis} with size {axis_len}");
        }
        return if i < 0 { b.ir_constant_int(i + len) } else { idx.clone() };
    }
    if let Some(position) = b.checked_position(idx, axis_len) {
        return position;
    }
    let wrapped = normalize_dynamic_index(b, idx, axis_len);
    let position = crate::optim::resolver::discharge_index_in_range(b, &wrapped, 0, len, op_name);
    b.remember_checked_position(idx, axis_len, &position);
    position
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

            value_id: ValueId::next(),
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
    fn a_static_arrays_cells_are_in_its_storage_without_ir() {
        let mut b = IRBuilder::new();
        let lst = list_of((0..4).map(|i| int_value(&mut b, i)).collect());
        let before = b.stmts.len();
        let sa = to_static_array(&mut b, &lst).unwrap();
        let segment_id = match &sa {
            Value::StaticArray { segment_id, .. } => *segment_id,
            _ => panic!("expected StaticArray"),
        };
        assert_eq!(b.stmts.len(), before, "no IR for known cells");
        let cells = b.storage_cells(segment_id);
        assert_eq!(cells.iter().map(Value::int_val).collect::<Vec<_>>(), vec![Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(b.stmts.len(), before, "reading known cells needs no IR");
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
