//! Typed access to array storages (`builder/store.rs`) for the array
//! representations that address their cells by number: dynamic ndarrays
//! (every one owns a storage of its capacity) and the static arrays' memory
//! paths.
//!
//! A complex array is a float storage of (real, imaginary) pairs: element
//! `a` is the cells `2a` and `2a + 1`.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::scalar_i64_to_value;
use crate::types::{NumberType, ScalarValue, Value};

/// A new storage holding `values` (the legacy `ScalarValue<i64>` cells of a
/// dynamic ndarray, of `dtype`). Returns its id.
pub fn alloc_and_write(b: &mut IRBuilder, values: &[ScalarValue<i64>], dtype: NumberType) -> u32 {
    let values: Vec<Value> = values.iter().map(|elem| scalar_i64_to_value(elem, dtype)).collect();
    alloc_and_write_values(b, &values, dtype)
}

/// A new storage holding `values` (of `dtype`). Returns its id.
pub fn alloc_and_write_values(b: &mut IRBuilder, values: &[Value], dtype: NumberType) -> u32 {
    let cells: Vec<Value> = match dtype {
        NumberType::Complex => values
            .iter()
            .flat_map(|v| {
                let (re, im) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, v);
                [re, im]
            })
            .collect(),
        NumberType::Integer | NumberType::Float => values.to_vec(),
    };
    let storage = b.alloc_storage(cells, dtype);
    b.note_segment_cells(storage, values);
    storage
}

/// Read all `max_len` cells of a storage.
pub fn read_all(b: &mut IRBuilder, segment_id: u32, max_len: usize) -> Vec<Value> {
    (0..max_len).map(|i| b.storage_read(segment_id, i)).collect()
}

/// Read one element of a storage as a value of `dtype` (a complex element is
/// two cells).
pub fn read_typed(b: &mut IRBuilder, segment_id: u32, address: &Value, dtype: NumberType) -> Value {
    if dtype == NumberType::Complex {
        let two = b.ir_constant_int(2);
        let one = b.ir_constant_int(1);
        let real_addr = b.ir_mul_i(address, &two);
        let imag_addr = b.ir_add_i(&real_addr, &one);
        let real_raw = b.storage_read_at(segment_id, &real_addr);
        let imag_raw = b.storage_read_at(segment_id, &imag_addr);
        let real = crate::ops::dyn_ndarray::retag(&real_raw, NumberType::Float);
        let imag = crate::ops::dyn_ndarray::retag(&imag_raw, NumberType::Float);
        return crate::helpers::value_ops::pack_complex_value(real, imag);
    }
    let raw = b.storage_read_at(segment_id, address);
    crate::ops::dyn_ndarray::retag(&raw, dtype)
}

/// Write one element of a storage (a complex element is two cells).
pub fn write_typed(b: &mut IRBuilder, segment_id: u32, address: &Value, value: &Value, dtype: NumberType) {
    if dtype == NumberType::Complex {
        let two = b.ir_constant_int(2);
        let one = b.ir_constant_int(1);
        let real_addr = b.ir_mul_i(address, &two);
        let imag_addr = b.ir_add_i(&real_addr, &one);
        let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, value);
        b.storage_write_at(segment_id, &real_addr, &real);
        b.storage_write_at(segment_id, &imag_addr, &imag);
        return;
    }
    b.storage_write_at(segment_id, address, value);
}

/// Read all `max_len` elements of a storage as values of `dtype`, in
/// storage order. A possibly non-contiguous view reads in logical order
/// with [`read_all_logical_typed`].
pub fn read_all_typed(b: &mut IRBuilder, segment_id: u32, max_len: usize, dtype: NumberType) -> Vec<Value> {
    (0..max_len)
        .map(|i| {
            let addr = b.ir_constant_int(i as i64);
            read_typed(b, segment_id, &addr, dtype)
        })
        .collect()
}

/// Read all `product(logical_shape)` elements of a possibly non-contiguous
/// view of a storage in logical row-major order: element `coords` is the
/// cell `logical_offset + Σ coords[i] · logical_strides[i]`.
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
        return read_all_typed(b, segment_id, total, dtype);
    }
    (0..total)
        .map(|flat| {
            let coords = crate::helpers::shape_arith::decode_coords(flat, logical_shape, &row_major);
            let abs: usize = logical_offset + coords.iter().zip(logical_strides).map(|(c, s)| c * s).sum::<usize>();
            let addr = b.ir_constant_int(abs as i64);
            read_typed(b, segment_id, &addr, dtype)
        })
        .collect()
}

pub fn read_all_complex_typed(b: &mut IRBuilder, segment_id: u32, _imag_segment_id: u32, max_len: usize) -> Vec<Value> {
    read_all_typed(b, segment_id, max_len, NumberType::Complex)
}
