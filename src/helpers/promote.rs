//! Static → dynamic ndarray promotion.
//!
//! The lazy automatic-promotion path takes a `Value::List`/`Value::Tuple`
//! that represents a static-shape ndarray and produces an equivalent
//! `Value::DynamicNDArray` with the tightest possible envelope (every dim
//! is statically known, `min == max == static_size`). This is the bridge
//! that lets static values flow into dynamic op surfaces transparently.
//!
//! Decisions baked in here:
//! - Per `ROADMAP/03-type-system.md` §3.6: lowering is one-way, lazy, and
//!   automatic; the runtime metadata is pinned to the static shape.
//! - No ZKRAM segment is allocated by this helper. Existing dynamic ops
//!   carry their payload as `Vec<ScalarValue<i64>>` in `elements`; the
//!   ZKRAM lowering happens in the backend. When the segment-based
//!   payload model lands (Phase 2+), this helper grows an
//!   `ir_allocate_memory` + `storage_write_at` step.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{
    Dim, DynArrayMeta, DynamicNDArrayData, Envelope, NumberType, ScalarValue, Value, ValueId,
};

use super::composite::{flatten_composite, get_composite_shape};
use super::shape_arith::row_major_strides;

/// Infer the dtype of a static composite by looking at its leaves. Any
/// `Complex` leaf forces `Complex`; any `Float` leaf forces `Float`;
/// otherwise `Integer`. A `StaticArray` input reports its own dtype.
fn infer_static_dtype(val: &Value) -> NumberType {
    if let Value::StaticArray { dtype, .. } = val {
        return *dtype;
    }
    let leaves = flatten_composite(val);
    if leaves.iter().any(|v| matches!(v, Value::Complex { .. })) {
        NumberType::Complex
    } else if leaves.iter().any(|v| matches!(v, Value::Float(_))) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

/// Promote a static composite ndarray (`Value::List`/`Value::Tuple` of
/// numeric leaves) into a `Value::DynamicNDArray` with the tightest
/// envelope (one `Static(N)` dim per axis). The runtime metadata is
/// initialised to match the static shape.
///
/// Scalars (non-composite numeric values) become rank-0 dynamic arrays.
///
/// Panics if `val` is not a static numeric value or composite of numeric
/// leaves.
/// Contract-boundary variant of [`promote_static_to_dynamic`]: a
/// segment-backed StaticArray with runtime-extents provenance (a
/// dynamic-bounds slice materialization) promotes with its TRUE runtime
/// shape — content occupies the `[0, extent_i)` prefix of each envelope
/// axis (padded layout). Without this, the promotion claims the full
/// envelope rectangle and the output contract leaks static skeleton
/// dims instead of the runtime extents (the 0507/0509/0510
/// binding-audit xfails).
///
/// ONLY for values that feed the output contract (circuit return
/// exposure, the if-merge select feeding a return). Mid-computation
/// coercions must use the plain promotion: switching them to runtime
/// dims flips every downstream op from static to dynamic lowering
/// (0389/0419/0450 regressed 30-600x in gates when this fired
/// unconditionally).
pub fn promote_static_to_dynamic_with_extents(b: &mut IRBuilder, val: &Value) -> Value {
    if let Value::StaticArray {
        segment_id,
        shape: sa_shape,
        strides: sa_strides,
        offset: 0,
        dtype: sa_dtype,
        ..
    } = val
    {
        if *sa_strides == row_major_strides(sa_shape) && *sa_dtype != NumberType::Complex {
            if let Some(ext_vids) = b.provenance.runtime_extents_of(*segment_id).cloned() {
                if ext_vids.len() == sa_shape.len() {
                    let mut ext_vals: Vec<Value> = Vec::with_capacity(ext_vids.len());
                    let mut ok = true;
                    for vid in &ext_vids {
                        match b.value_of_vid(*vid) {
                            Some(v) => ext_vals.push(v),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        let rank = sa_shape.len();
                        let total: usize = sa_shape.iter().product();
                        let mut dims = Vec::with_capacity(rank);
                        let mut rt_shape = Vec::with_capacity(rank);
                        let mut len_v: Option<Value> = None;
                        for (ax, ev) in ext_vals.iter().enumerate() {
                            if ev.int_val() == Some(sa_shape[ax] as i64) {
                                dims.push(Dim::new_static(&mut b.dim_table, sa_shape[ax]));
                            } else {
                                dims.push(Dim::new_dynamic(&mut b.dim_table, 0, sa_shape[ax]));
                            }
                            rt_shape.push(value_to_scalar_i64(ev));
                            len_v = Some(match len_v.take() {
                                None => ev.clone(),
                                Some(prev) => b.ir_mul_i(&prev, ev),
                            });
                        }
                        let runtime_length =
                            value_to_scalar_i64(&len_v.unwrap_or_else(|| b.ir_constant_int(0)));
                        let strides = row_major_strides(sa_shape);
                        return Value::DynamicNDArray(DynamicNDArrayData {
                            envelope: Envelope::new_with_bound(dims, total),
                            dtype: *sa_dtype,
                            segment_id: *segment_id,
                            meta: DynArrayMeta {
                                logical_shape: sa_shape.clone(),
                                logical_offset: 0,
                                logical_strides: strides.clone(),
                                runtime_length,
                                runtime_rank: ScalarValue::new(Some(rank as i64), None),
                                runtime_shape: rt_shape,
                                runtime_strides: strides
                                    .iter()
                                    .map(|&s| ScalarValue::new(Some(s as i64), None))
                                    .collect(),
                                runtime_offset: ScalarValue::new(Some(0), None),
                            },
                            value_id: ValueId::next(),
                        });
                    }
                }
            }
        }
    }
    promote_static_to_dynamic(b, val)
}

/// Promote a static composite ndarray (`Value::List`/`Value::Tuple` of
/// numeric leaves) into a `Value::DynamicNDArray` with the tightest
/// envelope (one `Static(N)` dim per axis). The runtime metadata is
/// initialised to match the static shape.
///
/// Scalars (non-composite numeric values) become rank-0 dynamic arrays.
///
/// Panics if `val` is not a static numeric value or composite of numeric
/// leaves.
pub fn promote_static_to_dynamic(b: &mut IRBuilder, val: &Value) -> Value {
    let shape = get_composite_shape(val);
    let dtype = infer_static_dtype(val);

    // Flatten leaves to the row-major payload representation used by
    // existing dynamic ops.
    let flat = flatten_composite(val);

    let envelope = Envelope::from_static_shape(&mut b.dim_table, &shape);
    let strides = row_major_strides(&shape);
    let total: usize = shape.iter().product();
    let rank = shape.len();

    let segment_id = if dtype == NumberType::Complex {
        // Complex leaves store as interleaved (real, imag) cell pairs —
        // `value_to_scalar_i64` of a Complex is a sourceless placeholder
        // and would crash the segment write.
        super::segment::alloc_and_write_values(b, &flat, dtype)
    } else {
        let elements: Vec<ScalarValue<i64>> = flat.iter().map(value_to_scalar_i64).collect();
        let segment_id = super::segment::alloc_and_write(b, &elements, dtype);
        // The scalar payload drops the `Boolean` tag of a bool array's cells.
        b.note_segment_cells(segment_id, &flat);
        segment_id
    };

    let dyn_data = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: shape.clone(),
            logical_offset: 0,
            logical_strides: strides.clone(),
            runtime_length: ScalarValue::new(Some(total as i64), None),
            runtime_rank: ScalarValue::new(Some(rank as i64), None),
            runtime_shape: shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    };

    Value::DynamicNDArray(dyn_data)
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositeData;
    use crate::types::ZinniaType;

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
    fn promote_1d_int_list() {
        let mut b = IRBuilder::new();
        let elems = (0..4).map(|i| int_value(&mut b, i)).collect();
        let lst = list_of(elems);
        let promoted = promote_static_to_dynamic(&mut b, &lst);
        let dyn_data = match &promoted {
            Value::DynamicNDArray(d) => d,
            _ => panic!("expected DynamicNDArray, got {:?}", promoted),
        };
        assert_eq!(dyn_data.envelope.rank(), 1);
        assert_eq!(dyn_data.envelope.is_fully_static(), Some(vec![4]));
        assert_eq!(dyn_data.envelope.max_total(), 4);
        assert_eq!(dyn_data.max_length(), 4);
        // segment_id is a valid u32 (0 is a valid first allocation)
        assert_eq!(dyn_data.dtype, NumberType::Integer);
        assert_eq!(dyn_data.meta.logical_shape, vec![4]);
        assert_eq!(dyn_data.meta.runtime_length.static_val, Some(4));
    }

    #[test]
    fn promote_2d_int_list() {
        let mut b = IRBuilder::new();
        let row0 = list_of((0..3).map(|i| int_value(&mut b, i)).collect());
        let row1 = list_of((10..13).map(|i| int_value(&mut b, i)).collect());
        let row2 = list_of((20..23).map(|i| int_value(&mut b, i)).collect());
        let mat = list_of(vec![row0, row1, row2]);
        let promoted = promote_static_to_dynamic(&mut b, &mat);
        let dyn_data = match &promoted {
            Value::DynamicNDArray(d) => d,
            _ => panic!("expected DynamicNDArray"),
        };
        assert_eq!(dyn_data.envelope.is_fully_static(), Some(vec![3, 3]));
        assert_eq!(dyn_data.envelope.max_total(), 9);
        assert_eq!(dyn_data.max_length(), 9);
        // segment_id is a valid u32 (0 is a valid first allocation)
        assert_eq!(dyn_data.meta.logical_strides, vec![3, 1]);
    }

    #[test]
    fn promote_dtype_inference_float() {
        let mut b = IRBuilder::new();
        let f = b.ir_constant_float(1.5);
        let i = b.ir_constant_int(2);
        let mixed = list_of(vec![f, i]);
        let promoted = promote_static_to_dynamic(&mut b, &mixed);
        let dyn_data = match &promoted {
            Value::DynamicNDArray(d) => d,
            _ => panic!("expected DynamicNDArray"),
        };
        assert_eq!(dyn_data.dtype, NumberType::Float);
    }

    #[test]
    fn promote_dtype_inference_integer() {
        let mut b = IRBuilder::new();
        let lst = list_of((0..3).map(|i| int_value(&mut b, i)).collect());
        let promoted = promote_static_to_dynamic(&mut b, &lst);
        let dyn_data = match &promoted {
            Value::DynamicNDArray(d) => d,
            _ => panic!("expected DynamicNDArray"),
        };
        assert_eq!(dyn_data.dtype, NumberType::Integer);
    }

    #[test]
    fn promote_envelope_dim_vars_are_unique() {
        let mut b = IRBuilder::new();
        // Allocate two arrays of the same shape and check that their
        // dim vars are different (no spurious unification).
        let a = list_of((0..4).map(|i| int_value(&mut b, i)).collect());
        let p1 = promote_static_to_dynamic(&mut b, &a);
        let p2 = promote_static_to_dynamic(&mut b, &a);
        let d1 = match &p1 {
            Value::DynamicNDArray(d) => d,
            _ => panic!(),
        };
        let d2 = match &p2 {
            Value::DynamicNDArray(d) => d,
            _ => panic!(),
        };
        assert_ne!(d1.envelope.dims[0].var, d2.envelope.dims[0].var);
    }

    #[test]
    fn promote_round_trip_via_value_zinnia_type() {
        // The `Value::zinnia_type()` accessor must still work after
        // migrating max_length / max_rank from fields to methods.
        let mut b = IRBuilder::new();
        let lst = list_of((0..6).map(|i| int_value(&mut b, i)).collect());
        let promoted = promote_static_to_dynamic(&mut b, &lst);
        let zt = promoted.zinnia_type();
        match zt {
            ZinniaType::DynamicNDArray {
                max_length,
                max_rank,
                ..
            } => {
                assert_eq!(max_length, 6);
                assert_eq!(max_rank, 1);
            }
            _ => panic!("expected DynamicNDArray ZinniaType, got {:?}", zt),
        }
    }
}
