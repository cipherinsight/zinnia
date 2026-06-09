//! ZKRAM segment helpers for dynamic ndarrays.
//!
//! Every `DynamicNDArrayData` is backed by a ZKRAM segment. Constructors
//! allocate a segment and write initial values via [`alloc_and_write`].
//! Reads go through `b.ir_read_memory(segment_id, addr)`.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::{
    complex_parts_to_value, scalar_i64_to_value, value_to_complex_parts, value_to_scalar_i64,
};
use crate::types::{NumberType, ScalarValue, Value};

/// Allocate a fresh ZKRAM segment and write `values` into it. Returns the
/// segment ID. Used by constructors and producing ops to create the backing
/// store for a new `DynamicNDArrayData`.
pub fn alloc_and_write(b: &mut IRBuilder, values: &[ScalarValue<i64>], dtype: NumberType) -> u32 {
    let seg = b.alloc_segment_id();
    let storage_len = match dtype {
        NumberType::Complex => values.len().saturating_mul(2),
        NumberType::Integer | NumberType::Float => values.len(),
    };
    b.ir_allocate_memory(seg, storage_len as u32, 0);
    for (i, elem) in values.iter().enumerate() {
        let val = scalar_i64_to_value(elem, dtype);
        let addr = b.ir_constant_int(i as i64);
        write_typed(b, seg, &addr, &val, dtype);
    }
    seg
}

pub fn alloc_and_write_values(b: &mut IRBuilder, values: &[Value], dtype: NumberType) -> u32 {
    let seg = b.alloc_segment_id();
    let storage_len = match dtype {
        NumberType::Complex => values.len().saturating_mul(2),
        NumberType::Integer | NumberType::Float => values.len(),
    };
    b.ir_allocate_memory(seg, storage_len as u32, 0);
    // Bound-fact capture for downstream fact-aware reads happens inside
    // `b.ir_write_memory` (the chokepoint that sees both `addr.int_val()`
    // and the value's facts). No per-cell capture needed here.
    for (i, val) in values.iter().enumerate() {
        let addr = b.ir_constant_int(i as i64);
        write_typed(b, seg, &addr, val, dtype);
    }
    seg
}

#[allow(dead_code)]
pub fn alloc_and_write_complex(b: &mut IRBuilder, values: &[crate::types::Value]) -> u32 {
    alloc_and_write_values(b, values, NumberType::Complex)
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

/// Read one segment element and tag it with the logical element dtype.
///
/// `IR::ReadMemory` is storage-typed as an integer field cell. Dynamic array
/// callers must restore the array element dtype at the IR value layer so later
/// float arithmetic does not depend on backend `FloatCast` guessing.
pub fn read_typed(b: &mut IRBuilder, segment_id: u32, address: &Value, dtype: NumberType) -> Value {
    if dtype == NumberType::Complex {
        let two = b.ir_constant_int(2);
        let one = b.ir_constant_int(1);
        let real_addr = b.ir_mul_i(address, &two);
        let imag_addr = b.ir_add_i(&real_addr, &one);
        let real_raw = b.ir_read_memory(segment_id, &real_addr);
        let imag_raw = b.ir_read_memory(segment_id, &imag_addr);
        let real = value_to_scalar_i64(&real_raw);
        let imag = value_to_scalar_i64(&imag_raw);
        return complex_parts_to_value(&real, &imag);
    }
    let raw = b.ir_read_memory(segment_id, address);
    scalar_i64_to_value(&crate::ops::dyn_ndarray::value_to_scalar_i64(&raw), dtype)
}

pub fn write_typed(
    b: &mut IRBuilder,
    segment_id: u32,
    address: &Value,
    value: &Value,
    dtype: NumberType,
) {
    if dtype == NumberType::Complex {
        let two = b.ir_constant_int(2);
        let one = b.ir_constant_int(1);
        let real_addr = b.ir_mul_i(address, &two);
        let imag_addr = b.ir_add_i(&real_addr, &one);
        let (real, imag) = value_to_complex_parts(value);
        b.ir_write_memory(
            segment_id,
            &real_addr,
            &scalar_i64_to_value(&real, NumberType::Float),
        );
        b.ir_write_memory(
            segment_id,
            &imag_addr,
            &scalar_i64_to_value(&imag, NumberType::Float),
        );
        return;
    }
    b.ir_write_memory(segment_id, address, value);
}

/// Read all `max_len` elements from a segment and tag each value with dtype.
///
/// Reads in **segment storage order** (`addr = 0..max_len`). Callers that
/// need logical-row-major order on a possibly-non-contiguous view (e.g. a
/// stride-permuted transpose view that reuses the source segment) should
/// use [`read_all_logical_typed`] instead.
pub fn read_all_typed(
    b: &mut IRBuilder,
    segment_id: u32,
    max_len: usize,
    dtype: NumberType,
) -> Vec<Value> {
    (0..max_len)
        .map(|i| {
            let addr = b.ir_constant_int(i as i64);
            read_typed(b, segment_id, &addr, dtype)
        })
        .collect()
}

/// Read all `total = product(logical_shape)` elements of a possibly-non-
/// contiguous DynamicNDArray view in **logical-row-major order**.
///
/// For a contiguous source (logical_strides == row_major_strides(logical_shape)
/// and logical_offset == 0), behaves identically to [`read_all_typed`] over
/// the first `total` segment slots, but accepts the larger `storage_len` so
/// non-contiguous views with the same segment but different total<max_len
/// still work.
///
/// For a non-contiguous view (e.g. stride-permuted transpose), walks the
/// logical row-major iteration order and computes the segment address for
/// each cell via `logical_offset + sum(coords[i] * logical_strides[i])`.
pub fn read_all_logical_typed(
    b: &mut IRBuilder,
    segment_id: u32,
    logical_shape: &[usize],
    logical_strides: &[usize],
    logical_offset: usize,
    dtype: NumberType,
) -> Vec<Value> {
    let total: usize = logical_shape.iter().product();
    let row_major = crate::helpers::shape_arith::row_major_strides(logical_shape);
    if logical_offset == 0 && logical_strides == row_major.as_slice() {
        // Fast path: contiguous from offset 0.
        return read_all_typed(b, segment_id, total, dtype);
    }
    let logical_row_strides = row_major;
    let mut out = Vec::with_capacity(total);
    for flat in 0..total {
        let coords =
            crate::helpers::shape_arith::decode_coords(flat, logical_shape, &logical_row_strides);
        let mut abs: usize = logical_offset;
        for (ax, &c) in coords.iter().enumerate() {
            abs += c * logical_strides[ax];
        }
        let addr = b.ir_constant_int(abs as i64);
        out.push(read_typed(b, segment_id, &addr, dtype));
    }
    out
}

pub fn read_all_complex_typed(
    b: &mut IRBuilder,
    segment_id: u32,
    _imag_segment_id: u32,
    max_len: usize,
) -> Vec<Value> {
    (0..max_len)
        .map(|i| {
            let addr = b.ir_constant_int(i as i64);
            read_typed(b, segment_id, &addr, NumberType::Complex)
        })
        .collect()
}
