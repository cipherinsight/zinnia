//! StaticArray implementations of `np.sort`, `np.argsort`, and axis-aware
//! `np.take` for common fixed-shape NumPy programs.
//!
//! A sort emits one `IR::Sorted` placeholder per output; the late pass
//! `lower::sort` lowers the live ones (a sorting network or pinned
//! prover hints, by a cost rule).

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::{decode_coords, encode_coords, row_major_strides};
use crate::types::{NumberType, SliceIndex, Value};

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
        // Materialized once, so every output of this sort names the same
        // element statements (that is how the lowering groups them).
        let fiber: std::sync::Arc<[Value]> = (0..axis_len)
            .map(|j| b.ensure_ptr(&cells[base + j * axis_stride]))
            .collect();
        // One placeholder per output; `lower::sort` lowers the
        // outputs that remain live after dead-code elimination.
        for rank in 0..axis_len {
            out[base + rank * axis_stride] = b.ir_sorted(&fiber, rank as u32, argsort);
        }
        let line: Vec<Value> = (0..axis_len).map(|rank| out[base + rank * axis_stride].clone()).collect();
        state_sorted_line_facts(b, &line, &fiber, argsort);
    }

    let result = build_static_array_from_flat(b, out, shape, out_dtype);
    // A sort of a bool array is a bool array (the placeholders are integers).
    if !argsort && crate::helpers::array_ops::is_bool_array(b, arr) {
        b.mark_bool_array(&result);
    }
    Some(result)
}

/// Array facts of one sorted line (`docs/array-facts.md`). Every
/// construction of the sort family yields NumPy's stable sort of the line on
/// every accepted witness (`lower::sort`, each construction's soundness
/// argument; the `Sort` lanes carry no execution predicate), so on every
/// path:
///
/// * argsort: the lanes are a permutation of `0..n` (hence in `[0, n - 1]`);
/// * sort: the lanes are non-decreasing, and each is one of the inputs, so
///   it lies in the inputs' range (derived from what is known of them).
///
/// The facts are about the line's cells: an N-D sort along an axis orders
/// each line, not the array (row-major order crosses lines).
fn state_sorted_line_facts(b: &mut IRBuilder, line: &[Value], inputs: &[Value], argsort: bool) {
    use crate::optim::array_facts::{ElemFacts, Order, Range, Segmentation};
    let n = line.len();
    if n == 0 {
        return;
    }
    if argsort {
        let facts = Segmentation::uniform(n, ElemFacts::range(Range::new(0, n as i64 - 1))).with_permutation();
        b.state_defined_facts(line, facts);
        return;
    }
    let order = ElemFacts { order: Some(Order::NonDecreasing), ..ElemFacts::TOP };
    b.state_defined_facts(line, Segmentation::uniform(n, order));
    let inputs = inputs.to_vec();
    b.state_derived_facts(line, |b, at| {
        let range = b.join_of(&inputs, at);
        (!range.is_top()).then(|| Segmentation::uniform(n, ElemFacts::range(range)))
    });
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

/// Whether every entry of a `np.take` index (a scalar, list or static array)
/// is a compile-time constant.
fn indices_are_constant(b: &mut IRBuilder, indices: &Value) -> bool {
    let entries = match indices {
        Value::StaticArray { .. } => payload_cells(b, indices),
        Value::List(_) | Value::Tuple(_) => crate::helpers::composite::flatten_composite(indices),
        _ => vec![indices.clone()],
    };
    entries.iter().all(|v| v.int_val().is_some())
}

/// `np.take(arr, indices, axis)` for runtime indices: NumPy's
/// `arr[:, ..., :, indices]` with `indices` on `axis` (`mode="raise"`: each
/// index wrapped once and checked, `checked_axis_index`). Bool entries are
/// positions 0 and 1 (`take` reads its indices as integers).
fn take_runtime_indices(b: &mut IRBuilder, arr: &Value, indices: &Value, axis: usize) -> Value {
    let positions = match indices {
        Value::StaticArray { .. } | Value::List(_) | Value::Tuple(_) => {
            let list = super::base::to_value_list(b, indices);
            let entries = crate::helpers::composite::flatten_composite(&list);
            let shape = crate::helpers::composite::get_composite_shape(&list);
            let entries: Vec<Value> = entries.iter().map(|v| b.int_dtype_cell(v)).collect();
            super::base::build_static_array_from_flat(b, entries, shape, NumberType::Integer)
        }
        Value::Boolean(_) => b.ir_int_cast(indices),
        _ => indices.clone(),
    };
    let mut index = vec![SliceIndex::Range(None, None, None); axis];
    index.push(SliceIndex::Single(positions));
    super::read::static_array_subscript(b, arr, &index)
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

    if !indices_are_constant(b, indices) {
        return Some(take_runtime_indices(b, arr, indices, axis));
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
    crate::optim::resolver::relay_forall_eq_const_from_input(b, &arr, &result);
    Some(result)
}
