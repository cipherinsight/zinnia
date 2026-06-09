//! StaticArray implementations of `np.sort`, `np.argsort`, and axis-aware
//! `np.take` for common fixed-shape NumPy programs.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, encode_coords, row_major_strides};
use crate::types::{NumberType, Value};

use super::base::build_static_array_from_flat;
use super::elementwise::payload_cells;

fn dtype_of(arr: &Value) -> NumberType {
    match arr {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => panic!("expected StaticArray"),
    }
}

fn shape_of(arr: &Value) -> Vec<usize> {
    match arr {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => panic!("expected StaticArray"),
    }
}

fn normalize_sort_axis(axis_arg: Option<&Value>, rank: usize, op: &str) -> Option<usize> {
    match axis_arg {
        None => Some(rank - 1),
        Some(Value::None) => None,
        Some(v) => {
            let raw = v
                .int_val()
                .unwrap_or_else(|| panic!("{}: axis must be a static integer or None", op));
            Some(crate::helpers::shape_arith::resolve_axis(raw, rank, op))
        }
    }
}

fn less_than(b: &mut IRBuilder, dtype: NumberType, lhs: &Value, rhs: &Value) -> Value {
    if dtype == NumberType::Float {
        b.ir_less_than_f(lhs, rhs)
    } else {
        b.ir_less_than_i(lhs, rhs)
    }
}

fn select_cell(
    b: &mut IRBuilder,
    dtype: NumberType,
    cond: &Value,
    tv: &Value,
    fv: &Value,
) -> Value {
    if dtype == NumberType::Float {
        b.ir_select_f(cond, tv, fv)
    } else {
        b.ir_select_i(cond, tv, fv)
    }
}

fn stable_sort_fiber(
    b: &mut IRBuilder,
    dtype: NumberType,
    values: &[Value],
) -> (Vec<Value>, Vec<Value>) {
    let n = values.len();
    let mut sorted = values.to_vec();
    let mut indices: Vec<Value> = (0..n).map(|i| b.ir_constant_int(i as i64)).collect();

    // Stable adjacent-swap network: swap only when right < left, so equal
    // values retain original index order for argsort ties. After outer pass
    // k, the largest k+1 elements have sunk to the tail, so the inner pass
    // only needs n-1-k comparisons. Total: n*(n-1)/2.
    for k in 0..n.saturating_sub(1) {
        for i in 0..n.saturating_sub(1 + k) {
            let swap = less_than(b, dtype, &sorted[i + 1], &sorted[i]);
            let left_v = select_cell(b, dtype, &swap, &sorted[i + 1], &sorted[i]);
            let right_v = select_cell(b, dtype, &swap, &sorted[i], &sorted[i + 1]);
            let left_i = b.ir_select_i(&swap, &indices[i + 1], &indices[i]);
            let right_i = b.ir_select_i(&swap, &indices[i], &indices[i + 1]);
            sorted[i] = left_v;
            sorted[i + 1] = right_v;
            indices[i] = left_i;
            indices[i + 1] = right_i;
        }
    }

    (sorted, indices)
}

pub fn try_apply_sort(
    b: &mut IRBuilder,
    arr: &Value,
    axis_arg: Option<&Value>,
    argsort: bool,
) -> Option<Value> {
    if !matches!(arr, Value::StaticArray { .. }) {
        return None;
    }
    let dtype = dtype_of(arr);
    if dtype == NumberType::Complex {
        panic!(
            "np.{}: Complex StaticArray sorting is not yet implemented",
            if argsort { "argsort" } else { "sort" }
        );
    }
    let op = if argsort { "np.argsort" } else { "np.sort" };
    let shape = shape_of(arr);
    if shape.is_empty() {
        panic!("{}: scalar StaticArray sorting is not supported", op);
    }
    let axis = match normalize_sort_axis(axis_arg, shape.len(), op) {
        Some(axis) => axis,
        None if shape.len() == 1 => 0,
        None => panic!("{} with axis=None flattening is not yet implemented", op),
    };

    let cells = payload_cells(b, arr);
    let strides = row_major_strides(&shape);
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let mut out = vec![b.ir_constant_int(0); cells.len()];
    let out_dtype = if argsort { NumberType::Integer } else { dtype };

    let kept_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &dim)| if i == axis { None } else { Some(dim) })
        .collect();
    let kept_strides = row_major_strides(&kept_shape);
    let kept_total = if kept_shape.is_empty() {
        1
    } else {
        kept_shape.iter().product()
    };

    for kept_idx in 0..kept_total {
        let kept_coords = if kept_shape.is_empty() {
            Vec::new()
        } else {
            decode_coords(kept_idx, &kept_shape, &kept_strides)
        };
        let mut coords = vec![0usize; shape.len()];
        let mut k = 0;
        for dim in 0..shape.len() {
            if dim != axis {
                coords[dim] = kept_coords[k];
                k += 1;
            }
        }
        let base = encode_coords(&coords, &strides);
        let fiber: Vec<Value> = (0..axis_len)
            .map(|j| cells[base + j * axis_stride].clone())
            .collect();
        let (sorted, indices) = stable_sort_fiber(b, dtype, &fiber);
        let selected = if argsort { indices } else { sorted };
        for (j, cell) in selected.into_iter().enumerate() {
            out[base + j * axis_stride] = cell;
        }
    }

    let result = build_static_array_from_flat(b, out, shape, out_dtype);
    // The available `is_sorted` vocabulary is whole-array/flattened. Axis-local
    // N-D sort is only row/column-local, so do not claim that global fact.
    if !argsort && result_shape_is_globally_sorted(&result) {
        if let Some(vid) = result.value_id() {
            b.fire_contract("sort_is_sorted", vid, &HashMap::new());
        }
    }
    Some(result)
}

fn result_shape_is_globally_sorted(result: &Value) -> bool {
    matches!(result, Value::StaticArray { shape, .. } if shape.len() == 1)
}

fn flatten_static_indices(b: &mut IRBuilder, indices: &Value) -> (Vec<i64>, Vec<usize>) {
    match indices {
        Value::Integer(_) => (
            vec![indices
                .int_val()
                .expect("np.take: scalar index must be statically known")],
            Vec::new(),
        ),
        Value::List(d) | Value::Tuple(d) => {
            let mut vals = Vec::with_capacity(d.values.len());
            for v in &d.values {
                vals.push(
                    v.int_val()
                        .expect("np.take: indices must be statically-known integers"),
                );
            }
            (vals, vec![d.values.len()])
        }
        Value::StaticArray { dtype, shape, .. } => {
            if *dtype != NumberType::Integer {
                panic!("np.take: indices array must have integer dtype");
            }
            let cells = payload_cells(b, indices);
            let mut vals = Vec::with_capacity(cells.len());
            for v in &cells {
                vals.push(
                    v.int_val()
                        .expect("np.take: indices must be statically-known integers"),
                );
            }
            (vals, shape.clone())
        }
        _ => panic!("np.take: indices must be a static integer or static integer array"),
    }
}

pub fn try_apply_take(
    b: &mut IRBuilder,
    arr: &Value,
    indices: &Value,
    axis_arg: Option<&Value>,
) -> Option<Value> {
    if !matches!(arr, Value::StaticArray { .. }) {
        return None;
    }
    let dtype = dtype_of(arr);
    if dtype == NumberType::Complex {
        panic!("np.take: Complex StaticArray take is not yet implemented");
    }
    let shape = shape_of(arr);
    if shape.is_empty() {
        panic!("np.take: scalar StaticArray input is not supported");
    }
    let axis_is_default = matches!(axis_arg, None | Some(Value::None));
    let axis = match axis_arg {
        None => 0,
        Some(Value::None) if shape.len() == 1 => 0,
        Some(Value::None) => {
            panic!("np.take on multi-dimensional arrays with axis=None (flatten) is not yet implemented")
        }
        Some(v) => {
            let raw = v
                .int_val()
                .expect("np.take: axis must be a statically-known integer or None");
            crate::helpers::shape_arith::resolve_axis(raw, shape.len(), "np.take")
        }
    };
    if axis_is_default && shape.len() != 1 {
        panic!(
            "np.take on multi-dimensional arrays with axis=None (flatten) is not yet implemented"
        );
    }

    let (raw_indices, index_shape) = flatten_static_indices(b, indices);
    let axis_len = shape[axis] as i64;
    let normalized: Vec<usize> = raw_indices
        .into_iter()
        .map(|i| {
            let idx = if i < 0 { axis_len + i } else { i };
            if idx < 0 || idx >= axis_len {
                panic!(
                    "np.take: index {} is out of bounds for axis {} with size {}",
                    i, axis, axis_len
                );
            }
            idx as usize
        })
        .collect();

    let mut out_shape = Vec::with_capacity(shape.len() - 1 + index_shape.len());
    out_shape.extend_from_slice(&shape[..axis]);
    out_shape.extend_from_slice(&index_shape);
    out_shape.extend_from_slice(&shape[axis + 1..]);

    let cells = payload_cells(b, arr);
    let src_strides = row_major_strides(&shape);
    let out_strides = row_major_strides(&out_shape);
    let out_total = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut out = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let out_coords = if out_shape.is_empty() {
            Vec::new()
        } else {
            decode_coords(out_idx, &out_shape, &out_strides)
        };
        let mut src_coords = vec![0usize; shape.len()];
        for dim in 0..axis {
            src_coords[dim] = out_coords[dim];
        }
        let index_rank = index_shape.len();
        let take_pos = if index_rank == 0 {
            0
        } else {
            let index_strides = row_major_strides(&index_shape);
            let index_coords = &out_coords[axis..axis + index_rank];
            encode_coords(index_coords, &index_strides)
        };
        src_coords[axis] = normalized[take_pos];
        for dim in axis + 1..shape.len() {
            src_coords[dim] = out_coords[dim - 1 + index_rank];
        }
        let src_idx = encode_coords(&src_coords, &src_strides);
        out.push(cells[src_idx].clone());
    }

    let result = build_static_array_from_flat(b, out, out_shape, dtype);
    if let (Some(in_vid), Some(out_vid)) = (arr.value_id(), result.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    Some(result)
}
