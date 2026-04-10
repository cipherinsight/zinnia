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
//!   `ir_allocate_memory` + `ir_write_memory` step.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{
    Dim, DynArrayMeta, DynamicNDArrayData, Envelope, NumberType, ScalarValue, Value,
};

use super::composite::{flatten_composite, get_composite_shape};
use super::shape_arith::row_major_strides;

/// Infer the dtype of a static composite by looking at its leaves. Any
/// `Float` leaf forces `Float`; otherwise `Integer`.
fn infer_static_dtype(val: &Value) -> NumberType {
    let leaves = flatten_composite(val);
    if leaves.iter().any(|v| matches!(v, Value::Float(_))) {
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
pub fn promote_static_to_dynamic(b: &mut IRBuilder, val: &Value) -> Value {
    let shape = get_composite_shape(val);
    let dtype = infer_static_dtype(val);

    // Flatten leaves to the row-major payload representation used by
    // existing dynamic ops.
    let flat = flatten_composite(val);
    let elements: Vec<ScalarValue<i64>> = flat.iter().map(value_to_scalar_i64).collect();

    let envelope = Envelope::from_static_shape(&mut b.dim_table, &shape);
    let strides = row_major_strides(&shape);
    let total: usize = shape.iter().product();
    let rank = shape.len();

    let mut dyn_data = DynamicNDArrayData {
        envelope,
        dtype,
        elements,
        segment_id: None,
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
    };

    // Materialize the elements into a ZKRAM segment so downstream dynamic
    // ops can read via ir_read_memory.
    super::segment::ensure_segment(b, &mut dyn_data);

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
        assert_eq!(dyn_data.elements.len(), 4);
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
        assert_eq!(dyn_data.elements.len(), 9);
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
            ZinniaType::DynamicNDArray { max_length, max_rank, .. } => {
                assert_eq!(max_length, 6);
                assert_eq!(max_rank, 1);
            }
            _ => panic!("expected DynamicNDArray ZinniaType, got {:?}", zt),
        }
    }
}
