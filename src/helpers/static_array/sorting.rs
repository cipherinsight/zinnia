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

fn equal(b: &mut IRBuilder, dtype: NumberType, lhs: &Value, rhs: &Value) -> Value {
    if dtype == NumberType::Float {
        b.ir_equal_f(lhs, rhs)
    } else {
        b.ir_equal_i(lhs, rhs)
    }
}

/// The compare-exchange pairs `(i, j)`, `i < j`, of Batcher's
/// merge-exchange sorting network for `n` keys, in execution order (Knuth,
/// TAOCP vol. 3, 5.2.2, Algorithm M). Applying "put the smaller key at
/// `i`" to every pair sorts any `n` keys. It uses O(n log^2 n)
/// comparators: 97 for n = 20, 395 for n = 50, 1,077 for n = 100, where an
/// adjacent-swap network needs n(n-1)/2 (190, 1,225, 4,950).
fn merge_exchange_pairs(n: usize) -> Vec<(usize, usize)> {
    if n < 2 {
        return Vec::new();
    }
    let t = usize::BITS - (n - 1).leading_zeros(); // ceil(log2 n)
    let mut pairs = Vec::new();
    let mut p = 1usize << (t - 1);
    while p > 0 {
        let (mut q, mut r, mut d) = (1usize << (t - 1), 0, p);
        loop {
            pairs.extend((0..n - d).filter(|i| i & p == r).map(|i| (i, i + d)));
            if q == p {
                break;
            }
            d = q - p;
            q >>= 1;
            r = p;
        }
        p >>= 1;
    }
    pairs
}

/// The compare-exchange pairs of the adjacent-swap (bubble) network: pass
/// `k` carries the largest remaining key to position `n - 1 - k`, so it
/// needs n(n-1)/2 comparators. Its comparators only exchange neighbours
/// that are strictly out of order, so equal keys never change their
/// relative order: the network is stable without any tie-break.
fn adjacent_swap_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n.saturating_sub(1))
        .flat_map(|k| (0..n - 1 - k).map(|i| (i, i + 1)))
        .collect()
}

/// Comparator cost of the argsort tie-break relative to a plain
/// compare-exchange: the equality test, the index comparison and the two
/// logic gates roughly double it (measured 1.8x on halo2-base, 1.2x on
/// Noir, 2026-10-06). Used only to pick the cheaper network.
const TIE_BREAK_COST_FACTOR: usize = 2;

/// Sorts `values` ascending; with `track_indices`, also returns each
/// output's original position (the stable argsort).
///
/// Values alone use Batcher's network ([`merge_exchange_pairs`]); equal
/// values are interchangeable in the output, so stability is moot. For
/// the argsort the order of equal values matters (NumPy's
/// `kind="stable"`). A sorting network is not stable by itself, so every
/// key becomes the pair `(value, original index)`, compared
/// lexicographically: these keys are distinct and their ascending order
/// is the stable order. The index comparison is `LtIBounded`, sound
/// because every index cell is a selection among the constants `0..n`.
/// When that tie-break makes Batcher's network costlier than the
/// adjacent-swap network (small `n`), the latter is used: it is stable
/// with plain comparators.
fn sort_fiber(
    b: &mut IRBuilder,
    dtype: NumberType,
    values: &[Value],
    track_indices: bool,
) -> (Vec<Value>, Vec<Value>) {
    let n = values.len();
    let batcher = merge_exchange_pairs(n);
    let adjacent_cost = n * n.saturating_sub(1) / 2;
    let (pairs, tie_break) = if !track_indices {
        (batcher, false)
    } else if batcher.len() * TIE_BREAK_COST_FACTOR < adjacent_cost {
        (batcher, true)
    } else {
        (adjacent_swap_pairs(n), false)
    };
    let mut keys = values.to_vec();
    let mut indices: Vec<Value> = if track_indices {
        (0..n).map(|i| b.ir_constant_int(i as i64)).collect()
    } else {
        Vec::new()
    };
    for (i, j) in pairs {
        let value_less = less_than(b, dtype, &keys[j], &keys[i]);
        let swap = if tie_break {
            let tie = equal(b, dtype, &keys[j], &keys[i]);
            let index_less = b.ir_less_than_i_bounded(&indices[j], &indices[i]);
            let tie_swap = b.ir_logical_and(&tie, &index_less);
            b.ir_logical_or(&value_less, &tie_swap)
        } else {
            value_less
        };
        let low = select_cell(b, dtype, &swap, &keys[j], &keys[i]);
        let high = select_cell(b, dtype, &swap, &keys[i], &keys[j]);
        keys[i] = low;
        keys[j] = high;
        if track_indices {
            let low = b.ir_select_i(&swap, &indices[j], &indices[i]);
            let high = b.ir_select_i(&swap, &indices[i], &indices[j]);
            indices[i] = low;
            indices[j] = high;
        }
    }
    (keys, indices)
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
        let (sorted, indices) = sort_fiber(b, dtype, &fiber, argsort);
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

#[cfg(test)]
mod tests {
    use super::merge_exchange_pairs;

    fn apply(pairs: &[(usize, usize)], keys: &mut [u32]) {
        for &(i, j) in pairs {
            assert!(i < j && j < keys.len());
            if keys[j] < keys[i] {
                keys.swap(i, j);
            }
        }
    }

    /// The 0-1 principle: a comparator network sorts every input iff it
    /// sorts every sequence of zeros and ones.
    #[test]
    fn merge_exchange_sorts_every_zero_one_input() {
        for n in 0..=14usize {
            let pairs = merge_exchange_pairs(n);
            for mask in 0u32..(1 << n) {
                let mut keys: Vec<u32> = (0..n).map(|i| (mask >> i) & 1).collect();
                apply(&pairs, &mut keys);
                assert!(keys.windows(2).all(|w| w[0] <= w[1]), "n={n} mask={mask:b}");
            }
        }
    }

    #[test]
    fn merge_exchange_sorts_permutations_of_larger_sizes() {
        for n in [20usize, 50, 100, 129] {
            let pairs = merge_exchange_pairs(n);
            for seed in 0..20u64 {
                // A deterministic pseudo-random permutation (LCG shuffle).
                let mut keys: Vec<u32> = (0..n as u32).collect();
                let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                for i in (1..n).rev() {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    keys.swap(i, (state >> 33) as usize % (i + 1));
                }
                apply(&pairs, &mut keys);
                assert_eq!(keys, (0..n as u32).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn merge_exchange_comparator_counts() {
        let counts: Vec<usize> = [2, 3, 20, 50, 100].iter().map(|&n| merge_exchange_pairs(n).len()).collect();
        assert_eq!(counts, vec![1, 3, 97, 395, 1077]);
    }
}
