//! ZKRAM segment helpers for dynamic ndarrays.
//!
//! Every `DynamicNDArrayData` is backed by a ZKRAM segment. Constructors
//! allocate a segment and write initial values via [`alloc_and_write`].
//! Reads go through `b.ir_read_memory(segment_id, addr)`.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::scalar_i64_to_value;
use crate::types::{NumberType, ScalarValue, Value};

/// Allocate a fresh ZKRAM segment and write `values` into it. Returns the
/// segment ID. Used by constructors and producing ops to create the backing
/// store for a new `DynamicNDArrayData`.
pub fn alloc_and_write(
    b: &mut IRBuilder,
    values: &[ScalarValue<i64>],
    dtype: NumberType,
) -> u32 {
    let seg = b.alloc_segment_id();
    b.ir_allocate_memory(seg, values.len() as u32, 0);
    for (i, elem) in values.iter().enumerate() {
        let val = scalar_i64_to_value(elem, dtype);
        let addr = b.ir_constant_int(i as i64);
        b.ir_write_memory(seg, &addr, &val);
    }
    seg
}

/// Read all `max_len` elements from a segment as `Value`s.
pub fn read_all(b: &mut IRBuilder, segment_id: u32, max_len: usize) -> Vec<Value> {
    (0..max_len)
        .map(|i| {
            let addr = b.ir_constant_int(i as i64);
            b.ir_read_memory(segment_id, &addr)
        })
        .collect()
}
