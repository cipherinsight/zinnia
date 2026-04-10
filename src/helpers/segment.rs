//! ZKRAM segment helpers for dynamic ndarrays.
//!
//! Bridges `DynamicNDArrayData`'s compile-time `elements` cache and the
//! halo2 backend's ZKRAM read/write traces. Two essential operations:
//!
//! - [`materialize_to_segment`]: allocate a segment and write cached
//!   elements into it via `ir_write_memory`.
//! - [`ensure_segment`]: lazy version — materializes only if the array
//!   doesn't already have a segment.
//!
//! Reading from a segment is a single `b.ir_read_memory(segment_id, addr)`
//! call on the builder — no wrapper needed.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::scalar_i64_to_value;
use crate::types::DynamicNDArrayData;

/// Materialize a dynamic ndarray's cached `elements` into a fresh ZKRAM
/// segment. Returns the newly-allocated segment ID.
pub fn materialize_to_segment(b: &mut IRBuilder, data: &DynamicNDArrayData) -> u32 {
    let seg = b.alloc_segment_id();
    let max_len = data.max_length();
    b.ir_allocate_memory(seg, max_len as u32, 0);

    for (i, elem) in data.elements.iter().enumerate() {
        let val = scalar_i64_to_value(elem, data.dtype);
        let addr = b.ir_constant_int(i as i64);
        b.ir_write_memory(seg, &addr, &val);
    }

    seg
}

/// Ensure a `DynamicNDArrayData` has a ZKRAM segment. If it already has
/// one, return it. Otherwise materialize and set `data.segment_id`.
pub fn ensure_segment(b: &mut IRBuilder, data: &mut DynamicNDArrayData) -> u32 {
    if let Some(seg) = data.segment_id {
        return seg;
    }
    let seg = materialize_to_segment(b, data);
    data.segment_id = Some(seg);
    seg
}
