//! ZKRAM segment helpers for dynamic ndarrays.
//!
//! These are the low-level plumbing between `DynamicNDArrayData`'s
//! compile-time `elements: Vec<ScalarValue<i64>>` and the halo2 backend's
//! ZKRAM segment read/write traces. The pattern is:
//!
//! - **Materialize**: allocate a segment, write the cached elements into it.
//!   After this the array is addressable via `ir_read_memory`. The
//!   `segment_id` field on `DynamicNDArrayData` is set.
//!
//! - **Read**: `ir_read_memory(segment_id, addr)` — thin wrapper that
//!   panics if the array hasn't been materialized.
//!
//! Every producing dynamic op should eventually call `materialize` on its
//! output. During the transition period only the ops that genuinely need
//! ZKRAM (because their output length is data-dependent) use these
//! helpers; the rest still carry their payload in the `elements` Vec.

use crate::builder::IRBuilder;
use crate::types::{DynamicNDArrayData, NumberType, ScalarValue, Value};

/// Materialize a dynamic ndarray's cached `elements` into a ZKRAM segment.
/// Returns the newly-allocated `segment_id`. The segment is initialized to
/// 0 and then overwritten element-by-element via `ir_write_memory`.
///
/// After this call, the array's `segment_id` should be set to
/// `Some(returned_id)` so subsequent readers know it's materialized.
pub fn materialize_to_segment(b: &mut IRBuilder, data: &DynamicNDArrayData) -> u32 {
    let seg = b.alloc_segment_id();
    let max_len = data.max_length();
    b.ir_allocate_memory(seg, max_len as u32, 0);

    for (i, elem) in data.elements.iter().enumerate() {
        let val = scalar_to_value(b, elem, data.dtype);
        let addr = b.ir_constant_int(i as i64);
        b.ir_write_memory(seg, &addr, &val);
    }

    seg
}

/// Read a single element from a materialized segment.
pub fn read_segment(b: &mut IRBuilder, segment_id: u32, addr: &Value) -> Value {
    b.ir_read_memory(segment_id, addr)
}

/// Allocate a fresh output segment of the given size, initialized to 0.
/// Returns the segment ID.
pub fn allocate_output_segment(b: &mut IRBuilder, max_len: usize) -> u32 {
    let seg = b.alloc_segment_id();
    b.ir_allocate_memory(seg, max_len as u32, 0);
    seg
}

/// Ensure a `DynamicNDArrayData` has a ZKRAM segment. If it already has
/// one, return it. Otherwise, materialize the cached elements and return
/// the new segment ID.
///
/// This is the recommended entry point: callers don't have to care
/// whether the array was previously materialized or not.
pub fn ensure_segment(b: &mut IRBuilder, data: &mut DynamicNDArrayData) -> u32 {
    if let Some(seg) = data.segment_id {
        return seg;
    }
    let seg = materialize_to_segment(b, data);
    data.segment_id = Some(seg);
    seg
}

/// Convert a `ScalarValue<i64>` back to a `Value`, using the dtype to pick
/// `ir_constant_int` vs `ir_constant_float`. If the ScalarValue has a
/// static_val, emit a constant; otherwise this is a forward reference to a
/// previously-computed IR value (its `ptr` field, if any, carries the
/// statement ID).
fn scalar_to_value(b: &mut IRBuilder, sv: &ScalarValue<i64>, dtype: NumberType) -> Value {
    // If the ScalarValue has a known pointer (from a previous IR statement),
    // reconstruct the Value from that pointer. Otherwise emit a constant.
    if let Some(ptr) = sv.ptr {
        // Reconstruct — the ptr is an IR statement ID.
        match dtype {
            NumberType::Integer => Value::Integer(ScalarValue {
                static_val: sv.static_val,
                ptr: Some(ptr),
            }),
            NumberType::Float => Value::Float(crate::types::ScalarValue {
                static_val: sv.static_val.map(|v| v as f64),
                ptr: Some(ptr),
            }),
        }
    } else if let Some(val) = sv.static_val {
        match dtype {
            NumberType::Integer => b.ir_constant_int(val),
            NumberType::Float => b.ir_constant_float(val as f64),
        }
    } else {
        // No static val and no ptr — shouldn't happen in well-formed data.
        // Fall back to zero.
        match dtype {
            NumberType::Integer => b.ir_constant_int(0),
            NumberType::Float => b.ir_constant_float(0.0),
        }
    }
}
