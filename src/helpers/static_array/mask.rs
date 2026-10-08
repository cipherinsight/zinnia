//! Native boolean-mask read / write paths for `Value::StaticArray`.
//!
//! P5b of `compiler.epic-segment-native-static-arrays`: this module adds a
//! first-class boolean-masking implementation that operates directly on the
//! cached payload of segment-backed StaticArrays, avoiding the legacy
//! `boolean_mask_static` / `dyn_filter` round-trip through `Value::List`.
//!
//! ## Output type contract
//!
//! - **Static mask** (every cached cell carries a compile-time
//!   `static_val` — typically a constructor like `np.array([True, False,
//!   True])` or a comparison whose operands are constant): the surviving
//!   cell count is known at compile time, so the output is a fresh
//!   `Value::StaticArray` with `shape == [true_count]`. Cells of the
//!   surviving positions are placed in the new payload in original order.
//!
//! - **Dynamic mask** (one or more mask cells lack a compile-time
//!   `static_val` — e.g. comparison against a runtime input): the surviving
//!   cell count is data-dependent, so the output is a `Value::DynamicNDArray`
//!   with `total_bound == prod(arr.shape)`. The segment is built via the
//!   same compaction trick used by `dyn_filter` (write-pointer advances
//!   only on truthy mask cells; non-kept positions leave a placeholder
//!   that will either be overwritten by the next kept element or sit
//!   beyond `runtime_length`).
//!
//! ## Boolean-mask write
//!
//! `arr[mask] = v` is a per-cell `select(mask[i], v_i, arr[i])` that writes
//! the result back into `arr.segment_id` at offset `i`. RHS may be a
//! scalar (broadcast across all true positions) or another StaticArray /
//! List with as many leaves as the array has cells (numpy semantics: the
//! RHS for a boolean-mask write is sized to `true_count`, but for the
//! static-shape variant we accept either a same-shape RHS — common from
//! `arr[mask] = arr2[mask]` — or a true-count-sized RHS, broadcasting via
//! a running selection counter).
//!
//! Cache invalidation mirrors P3 — the mask write touches a runtime
//! pattern of cells, so the segment's cache entry is dropped (option (a)
//! from the P3 card).

use crate::builder::{IRBuilder, Where};
use crate::optim::array_facts::{ElemFacts, Range, Segmentation};
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ValueId};

use super::elementwise::payload_cells;

// ────────────────────────────────────────────────────────────────────────
// Mask classification
// ────────────────────────────────────────────────────────────────────────

/// Returns `true` when every cell carries a compile-time `static_val`. For a
/// Boolean StaticArray that means `bool_val()` is `Some(_)` for every cached
/// cell. For an Integer-storage Boolean (e.g. P4a comparison output whose
/// cells are still Boolean wires) we accept either `bool_val()` or
/// `int_val()`.
fn mask_is_fully_static(cells: &[Value]) -> bool {
    cells.iter().all(|c| match c {
        Value::Boolean(s) => s.static_val.is_some(),
        Value::Integer(s) => s.static_val.is_some(),
        _ => false,
    })
}

/// The compile-time truth of a mask cell, if it has one.
fn cell_static_truth_opt(c: &Value) -> Option<bool> {
    c.bool_val().or_else(|| c.int_val().map(|v| v != 0))
}

/// One cell of a masked write as an `OpKind::MaskWrite` placeholder (own
/// operands `[old, written, mask]`, see `lower::scatter::mask_write`).
fn mask_write_lane(b: &mut IRBuilder, old: &Value, written: &Value, mask: &Value) -> Value {
    let op = crate::lower::ArrayOp { kind: crate::lower::OpKind::MaskWrite, shared: 0, output: 0, position: Some(0) };
    let shared: std::sync::Arc<[Value]> = std::sync::Arc::from(Vec::new());
    b.ir_array_op(op, &shared, &[old.clone(), written.clone(), mask.clone()])
}

/// Pull the truthy/falsy compile-time value of a cell. Caller must have
/// established that `mask_is_fully_static` returned true.
fn cell_static_truth(c: &Value) -> bool {
    match c {
        Value::Boolean(s) => s.static_val.unwrap(),
        Value::Integer(s) => s.static_val.unwrap() != 0,
        _ => panic!("cell_static_truth: caller must guarantee static cell"),
    }
}

// ────────────────────────────────────────────────────────────────────────
// Public entry points
// ────────────────────────────────────────────────────────────────────────

/// Try to apply a boolean-mask read on a `Value::StaticArray`.
///
/// Returns `Some(result)` when both `arr` and `mask` are StaticArrays of
/// matching shape AND the mask cells are Boolean wires. The result is
/// either a `Value::StaticArray` (static mask) or a `Value::DynamicNDArray`
/// (dynamic mask).
pub fn try_apply_boolean_mask_read(b: &mut IRBuilder, arr: &Value, mask: &Value) -> Option<Value> {
    let (arr_dtype, arr_shape, _arr_seg) = match arr {
        Value::StaticArray {
            dtype,
            shape,
            segment_id,
            ..
        } => (*dtype, shape.clone(), *segment_id),
        _ => return None,
    };
    let mask_shape = match mask {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => return None,
    };
    if !crate::helpers::array_ops::is_bool_array(b, mask) {
        return None;
    }
    // numpy boolean-mask read accepts a `mask_shape` that is a prefix of
    // `arr_shape`: the leading axes are *collapsed and filtered* by the
    // mask, leaving the trailing axes intact. Equal-shape is the special
    // case where the trailing tail is empty (output is 1-D).
    //
    //   arr  : (D0, D1, ..., D_{k-1}, T0, T1, ..., T_{m-1})
    //   mask : (D0, D1, ..., D_{k-1})
    //   out  : (true_count, T0, T1, ..., T_{m-1})
    //
    // where `true_count ∈ [0, prod(mask_shape)]`. Anything else is a
    // genuine shape mismatch.
    let k = mask_shape.len();
    if k == 0 || k > arr_shape.len() || arr_shape[..k] != mask_shape[..] {
        panic!(
            "boolean index did not match indexed array along dimensions; \
             array shape {:?}, mask shape {:?}",
            arr_shape, mask_shape
        );
    }
    // Complex mask read is rejected for now (there's no defined Complex
    // ordering / mask interaction in numpy beyond eq / ne, which already
    // produces a Boolean mask via the comparison path).
    if matches!(arr_dtype, NumberType::Complex) {
        return None;
    }

    let leading_total: usize = mask_shape.iter().product();
    let trailing_dims: Vec<usize> = arr_shape[k..].to_vec();
    let trailing_total: usize = trailing_dims.iter().product();

    let arr_cells = payload_cells(b, arr);
    let mask_cells = payload_cells(b, mask);
    debug_assert_eq!(arr_cells.len(), leading_total * trailing_total);
    debug_assert_eq!(mask_cells.len(), leading_total);

    if mask_is_fully_static(&mask_cells) {
        // Static fast path: surviving leading-block count is known at
        // compile time. Concatenate the kept trailing-blocks in order.
        let mut surviving: Vec<Value> = Vec::new();
        let mut kept_leading: usize = 0;
        for i in 0..leading_total {
            if cell_static_truth(&mask_cells[i]) {
                let base = i * trailing_total;
                surviving.extend_from_slice(&arr_cells[base..base + trailing_total]);
                kept_leading += 1;
            }
        }
        let mut out_shape = vec![kept_leading];
        out_shape.extend_from_slice(&trailing_dims);
        return Some(super::base::build_static_array_from_flat(
            b, surviving, out_shape, arr_dtype,
        ));
    }

    // Dynamic mask: produce a DynamicNDArray of shape `(true_count,
    // *trailing_dims)` with `total_bound == arr.size`. Every output cell is
    // one of the input cells or the padding 0 (the compaction network selects
    // among them in-circuit, `compact_by_mask`), so the output's cells lie in
    // the input's range joined with 0 (array facts, derived where the input's
    // range holds).
    let input_ranges = [Where::Everywhere, Where::Here].map(|at| b.array_range(arr, at));
    let keeps_bool = crate::helpers::array_ops::is_bool_array(b, arr);
    let out = dynamic_mask_read(
        b,
        &arr_cells,
        &mask_cells,
        arr_dtype,
        leading_total,
        &trailing_dims,
    );
    if let Value::DynamicNDArray(d) = &out {
        let len = d.max_length();
        b.state_derived_storage_facts(d.segment_id, |b, at| {
            let input = input_ranges[at as usize];
            let range = input.join(&crate::optim::array_facts::Range::point(0), &b.oracle(at));
            (!input.is_top()).then(|| Segmentation::uniform(len, ElemFacts::range(range)))
        });
    }
    // The kept elements keep their dtype (the compaction's selects do not).
    if let (true, Value::DynamicNDArray(d)) = (keeps_bool, &out) {
        b.mark_bool_segment(d.segment_id);
    }
    Some(out)
}

/// Try to apply a boolean-mask write on a `Value::StaticArray`.
///
/// Returns `Some(arr)` (the same StaticArray, with its segment mutated) when
/// `arr` is a StaticArray and `mask` is a matching-shape Boolean mask. RHS
/// may be a scalar (broadcast) or a StaticArray / List of cells.
///
/// Cache invalidation: the segment cache is dropped because the post-write
/// pattern is data-dependent (any cell could have been overwritten if
/// the mask is dynamic; even for a static mask the write changes a
/// compile-time-known subset of cells).
/// Element facts across a boolean-mask write (`arr[mask] = value`, or
/// `arr[mask] op= delta` after `+=` lowering): capture the array's and the
/// value's ranges before the write ([`capture_mask_write_pre_bounds`]), and
/// state their join on the written array's cells after it
/// ([`deposit_mask_write_post`]): each cell of the written array is its old
/// value or (an element of) the written one, the write rule of
/// `docs/array-facts.md`. Called from the IR-gen visitor's
/// setitem-on-StaticArray dispatch, so it covers both the fast path
/// (`try_apply_boolean_mask_write`) and the fallback `static_array_setitem`.
pub struct MaskWritePreBounds {
    /// Per context (`Where::Everywhere`, `Where::Here`): the array's range
    /// and the value's.
    ranges: [(Range, Range); 2],
}

pub fn capture_mask_write_pre_bounds(
    b: &mut IRBuilder,
    arr: &Value,
    value: &Value,
) -> MaskWritePreBounds {
    let nothing = MaskWritePreBounds { ranges: [(Range::TOP, Range::TOP); 2] };
    let Value::StaticArray { dtype: NumberType::Integer, .. } = arr else { return nothing };
    let ranges = [Where::Everywhere, Where::Here].map(|at| (b.array_range(arr, at), b.range_of(value, at)));
    MaskWritePreBounds { ranges }
}

pub fn deposit_mask_write_post(b: &mut IRBuilder, pre: MaskWritePreBounds, updated: &Value) {
    let Some(cells) = b.leaf_cells(updated) else { return };
    let len = cells.len();
    b.state_derived_facts(&cells, |b, at| {
        let (arr, val) = pre.ranges[at as usize];
        if arr.is_top() || val.is_top() {
            return None;
        }
        let joined = arr.join(&val, &b.oracle(at));
        (!joined.is_top()).then(|| Segmentation::uniform(len, ElemFacts::range(joined)))
    });
}

pub fn try_apply_boolean_mask_write(
    b: &mut IRBuilder,
    arr: &Value,
    mask: &Value,
    value: &Value,
) -> Option<Value> {
    let (arr_dtype, arr_shape, arr_seg, arr_strides, arr_offset, imag_seg) = match arr {
        Value::StaticArray {
            dtype,
            shape,
            segment_id,
            strides,
            offset,
            imag_segment_id,
            value_id: _,
        } => (
            *dtype,
            shape.clone(),
            *segment_id,
            strides.clone(),
            *offset,
            *imag_segment_id,
        ),
        _ => return None,
    };
    let mask_shape = match mask {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => return None,
    };
    if !crate::helpers::array_ops::is_bool_array(b, mask) {
        return None;
    }
    if arr_shape != mask_shape {
        panic!(
            "boolean index did not match indexed array along dimensions; \
             array shape {:?}, mask shape {:?}",
            arr_shape, mask_shape
        );
    }

    let arr_cells = payload_cells(b, arr);
    let mask_cells = payload_cells(b, mask);
    let total: usize = arr_shape.iter().product();
    // The storage address of the view's element `i` (row-major): the view's
    // offset and strides (a transposed or column view is not contiguous).
    let logical_strides = crate::helpers::shape_arith::row_major_strides(&arr_shape);
    let address_of = |i: usize| -> usize {
        let coords = crate::helpers::shape_arith::decode_coords(i, &arr_shape, &logical_strides);
        arr_offset + coords.iter().zip(&arr_strides).map(|(c, s)| c * s).sum::<usize>()
    };

    // Resolve RHS into a flat Vec<Value>. Two shapes are accepted:
    //   - scalar (one element, broadcast across all true positions)
    //   - same-shape array (one cell per arr cell — common in
    //     `arr[mask] = arr2[mask]` after the read fast-path, since the
    //     read may return a same-true-count array which we need to
    //     conditionally place back into arr).
    enum Rhs<'a> {
        Scalar(&'a Value),
        FullShape(Vec<Value>),
        // RHS values are PACKED (numpy semantics: `arr[mask] = rhs`
        // consumes rhs sequentially, one element per true mask cell).
        // Aligned at write time by a running prefix count of trues —
        // which constant-folds when the mask is static.
        Compacted(Vec<Value>),
    }
    let rhs = if value.is_number() {
        Rhs::Scalar(value)
    } else if let Value::StaticArray { shape: rs, .. } = value {
        let cells = payload_cells(b, value);
        if rs == &arr_shape {
            // Same-shape RHS: positional (`arr[mask] = arr2[mask]` after
            // the read fast-path on a fully-true mask, or np.where-style
            // gated assignment).
            Rhs::FullShape(cells)
        } else {
            Rhs::Compacted(cells)
        }
    } else if matches!(value, Value::List(_) | Value::Tuple(_)) {
        let leaves = super::super::composite::flatten_composite(value);
        if leaves.len() == total {
            Rhs::FullShape(leaves)
        } else {
            Rhs::Compacted(leaves)
        }
    } else if let Value::DynamicNDArray(d) = value {
        // Aug-assign desugar `a[mask] = a[mask] + v` puts a DynamicNDArray
        // on the RHS: the mask-read gather output, which is COMPACTED
        // (true-position values packed to the front, zero-padded). Cell i
        // of the target therefore takes rhs[#trues before i], NOT rhs[i] —
        // the positional mapping wrote padding into late true positions
        // (binding-audit case 0490).
        //
        // Round-trip cancellation: when the RHS carries a mask-read view
        // whose mask is THIS mask (the same cell values: the same SSA
        // values, or statements computing the same values, see
        // `IRBuilder::same_value`), the packed
        // invariant gives rhs[#trues before i] == view.sources[i] for
        // every true position — the write is positional after all, with
        // no unpack network needed.
        // Identity-view guard: the consumed value must be the packed
        // array itself, not a re-view of its segment (a slice/reverse/
        // reshape shares the segment but reorders or truncates lanes,
        // breaking the positional mapping). Flat 1-D meta + the original
        // runtime-length stmt pin the identity.
        let meta_identity = d.meta.logical_offset == 0
            && d.meta.runtime_offset.static_val == Some(0)
            && d.meta.logical_shape.as_slice() == [total]
            && d.meta.logical_strides.as_slice() == [1]
            && d.meta.runtime_length.stmt_id.is_some();
        let view_sources = b
            .provenance
            .mask_read_view(d.segment_id)
            .filter(|view| {
                meta_identity
                    && view.count_stmt == d.meta.runtime_length.stmt_id
                    && view.sources.len() == total
                    && view.mask_cells.len() == mask_cells.len()
                    && view
                        .mask_cells
                        .iter()
                        .zip(mask_cells.iter())
                        .all(|(a, c)| match (a.value_id(), c.value_id()) {
                            (Some(x), Some(y)) if x == y => true,
                            // The same mask written twice (`x[x > c] = x[x > c] - d`)
                            // is two sets of statements computing one value.
                            _ => b.same_value(a, c),
                        })
            })
            .map(|view| view.sources.clone());
        if let Some(sources) = view_sources {
            Rhs::FullShape(sources)
        } else {
            let env_total = d.envelope.total_bound;
            let mut cells = Vec::with_capacity(env_total);
            for i in 0..env_total {
                let addr = b.ir_constant_int(i as i64);
                cells.push(crate::helpers::segment::read_typed(
                    b,
                    d.segment_id,
                    &addr,
                    d.dtype,
                ));
            }
            Rhs::Compacted(cells)
        }
    } else {
        // Unknown RHS shape — punt to legacy path.
        return None;
    };
    // The prefix-count alignment below is O(total^2) selects for dynamic
    // masks; keep a sane ceiling (static masks fold to nothing).
    if matches!(rhs, Rhs::Compacted(_)) && !mask_is_fully_static(&payload_cells(b, mask)) && total > 4096 {
        return None;
    }

    // Static fast-path bookkeeping for the TrueCount RHS form: walk the
    // mask's static cells to align rhs cells to the right positions.
    let static_mask = mask_is_fully_static(&mask_cells);

    if matches!(arr_dtype, NumberType::Complex) {
        // Component-aware write across both segments.
        let imag_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
        let mut rhs_idx_static: usize = 0;
        for i in 0..total {
            let addr = b.ir_constant_int(address_of(i) as i64);
            let mask_val = &mask_cells[i];
            let mask_bool = ensure_bool(b, mask_val);
            // Determine the per-cell new value.
            let new_val: Value = match &rhs {
                Rhs::Scalar(v) => (*v).clone(),
                Rhs::FullShape(cells) => cells[i].clone(),
                Rhs::Compacted(cells) => {
                    // Complex keeps the static-mask-only alignment; a
                    // dynamic mask with a packed Complex RHS falls back
                    // to the legacy path.
                    if !static_mask {
                        return None;
                    }
                    let pick = if cell_static_truth(mask_val) {
                        let v = cells.get(rhs_idx_static).cloned();
                        rhs_idx_static += 1;
                        v
                    } else {
                        None
                    };
                    pick.unwrap_or_else(|| arr_cells[i].clone())
                }
            };
            let (new_re, new_im) =
                crate::helpers::value_ops::unpack_value_to_complex_parts(b, &new_val);
            let (cur_re, cur_im) =
                crate::helpers::value_ops::unpack_value_to_complex_parts(b, &arr_cells[i]);
            let sel_re = b.ir_select_f(&mask_bool, &new_re, &cur_re);
            let sel_im = b.ir_select_f(&mask_bool, &new_im, &cur_im);
            b.storage_write_at(arr_seg, &addr, &sel_re);
            b.storage_write_at(imag_seg, &addr, &sel_im);
        }
        return Some(arr.clone());
    }

    // Non-Complex (Integer / Float) per-cell select + write. Every cell of
    // the view is written at its constant address, so later reads see the
    // written values directly (a chain of masked writes over one cell is
    // then visible to the scatter family's lowering).
    let mut rhs_idx_static: usize = 0;
    // Running count of true mask cells before position i — the packed-RHS
    // read pointer. Constant-folds away entirely for static masks.
    let mut prefix = b.ir_constant_int(0);
    for i in 0..total {
        let addr = b.ir_constant_int(address_of(i) as i64);
        let mask_val = &mask_cells[i];
        let mask_bool = ensure_bool(b, mask_val);
        let new_raw: Value = match &rhs {
            Rhs::Scalar(v) => (*v).clone(),
            Rhs::FullShape(cells) => cells[i].clone(),
            Rhs::Compacted(cells) => {
                if static_mask {
                    let pick = if cell_static_truth(mask_val) {
                        let v = cells.get(rhs_idx_static).cloned();
                        rhs_idx_static += 1;
                        v
                    } else {
                        None
                    };
                    pick.unwrap_or_else(|| arr_cells[i].clone())
                } else {
                    // rhs[#trues before i], one-hot over the prefix
                    // counter (at most i trues can precede position i).
                    let mut acc = arr_cells[i].clone();
                    for (k, c) in cells.iter().enumerate().take(i + 1) {
                        let k_c = b.ir_constant_int(k as i64);
                        let eq = b.ir_equal_i(&prefix, &k_c);
                        let c_cast = cast_to_dtype(b, c, arr_dtype);
                        let a_cast = cast_to_dtype(b, &acc, arr_dtype);
                        acc = if matches!(arr_dtype, NumberType::Float) {
                            b.ir_select_f(&eq, &c_cast, &a_cast)
                        } else {
                            b.ir_select_i(&eq, &c_cast, &a_cast)
                        };
                    }
                    acc
                }
            }
        };
        // NumPy converts the assigned value to the array's dtype (a bool
        // array stores truth values).
        let new_raw = if b.is_bool_segment(arr_seg) {
            b.bool_dtype_cell(&new_raw)
        } else {
            new_raw
        };
        let new_cast = cast_to_dtype(b, &new_raw, arr_dtype);
        let cur_cast = cast_to_dtype(b, &arr_cells[i], arr_dtype);
        let selected = if b.is_bool_segment(arr_seg) || cell_static_truth_opt(&mask_bool).is_some() {
            if matches!(arr_dtype, NumberType::Float) {
                b.ir_select_f(&mask_bool, &new_cast, &cur_cast)
            } else {
                b.ir_select_i(&mask_bool, &new_cast, &cur_cast)
            }
        } else {
            // A runtime mask: one `OpKind::MaskWrite` lane, which the scatter
            // family lowers (a select, or with the other writes of a chain
            // over this cell a sum over the masks' atoms).
            mask_write_lane(b, &cur_cast, &new_cast, &mask_bool)
        };
        // Under a runtime execution predicate (`if c: arr[mask] = v`) the
        // store keeps `select(c, selected, old)`.
        b.storage_write_at(arr_seg, &addr, &selected);
        if matches!(rhs, Rhs::Compacted(_)) && !static_mask {
            let m_i = b.ir_int_cast(&mask_bool);
            prefix = b.ir_add_i(&prefix, &m_i);
        }
    }
    Some(arr.clone())
}

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn ensure_bool(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Boolean(_) => v.clone(),
        _ => b.ir_bool_cast(v),
    }
}

fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) {
                v.clone()
            } else {
                b.ir_float_cast(v)
            }
        }
        NumberType::Integer => {
            if matches!(v, Value::Integer(_)) {
                v.clone()
            } else if matches!(v, Value::Boolean(_)) {
                let casted = b.ir_int_cast(v);
                if let (Value::Integer(mut s), Some(bv)) = (casted.clone(), v.bool_val()) {
                    s.static_val = Some(if bv { 1 } else { 0 });
                    Value::Integer(s)
                } else {
                    casted
                }
            } else {
                b.ir_int_cast(v)
            }
        }
        NumberType::Complex => unreachable!("cast_to_dtype(Complex) used outside Complex path"),
    }
}

/// Build a DynamicNDArray output from a boolean-mask read where the mask is
/// not fully compile-time-known. Mirrors `dyn_filter`'s compaction trick.
///
/// `leading_total = prod(mask.shape)` rows are scanned; each kept row
/// contributes a contiguous block of `trailing_total = prod(trailing_dims)`
/// arr cells to the output. When `trailing_dims` is empty the output
/// collapses to 1-D and this matches the original equal-shape contract.
fn dynamic_mask_read(
    b: &mut IRBuilder,
    arr_cells: &[Value],
    mask_cells: &[Value],
    dtype: NumberType,
    leading_total: usize,
    trailing_dims: &[usize],
) -> Value {
    let trailing_total: usize = trailing_dims.iter().product::<usize>().max(1);
    let total: usize = leading_total * trailing_total;
    let default_val = match dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Complex => unreachable!("Complex mask read rejected upstream"),
    };
    // Cast every source cell up to `dtype` (Boolean cells in an Integer
    // array, etc.). These cast values double as the MaskReadView sources.
    let cast_sources: Vec<Value> = arr_cells
        .iter()
        .map(|src| cast_to_dtype(b, src, dtype))
        .collect();

    // Compact via the SHARED one-hot shift network (the single copy of the
    // triangular shift loop, also used by `dyn_filter`).
    let crate::ops::dyn_ndarray::memory_ops::Compaction {
        acc,
        kept_rows: write_row,
    } = crate::ops::dyn_ndarray::memory_ops::compact_by_mask(
        b,
        &cast_sources,
        mask_cells,
        &default_val,
        trailing_total,
    );

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &acc, dtype);

    // Whole-element mask (no trailing tail): deposit the mask-read view
    // so an `a[mask] op= v` round trip can cancel the shift network into
    // per-element selects at the write (see `MaskReadView`). Deposit
    // AFTER the alloc writes — `on_segment_write` eager-drops the record
    // on any later write to this segment.
    if trailing_total == 1 && cast_sources.len() == leading_total {
        b.provenance.set_mask_read_view(
            segment_id,
            crate::provenance::MaskReadView {
                mask_cells: mask_cells.to_vec(),
                sources: cast_sources,
                count_stmt: write_row.stmt_id(),
            },
        );
    }

    // Envelope: leading dim is dynamic in `[0, leading_total]`; trailing
    // dims keep their compile-time size.
    let mut dims = vec![crate::types::Dim::new_dynamic(
        &mut b.dim_table,
        0,
        leading_total,
    )];
    for &td in trailing_dims {
        dims.push(crate::types::Dim::new_static(&mut b.dim_table, td));
    }
    let envelope = crate::types::Envelope::new_with_bound(dims, total);

    let runtime_row_count = value_to_scalar_i64(&write_row);
    let runtime_length_val = if trailing_dims.is_empty() {
        write_row.clone()
    } else {
        let trailing_total_val = b.ir_constant_int(trailing_total as i64);
        b.ir_mul_i(&write_row, &trailing_total_val)
    };
    let runtime_length = value_to_scalar_i64(&runtime_length_val);

    // logical shape uses leading_total as the per-axis max (the dynamic
    // upper bound); trailing dims are their static sizes.
    let mut logical_shape = vec![leading_total];
    logical_shape.extend_from_slice(trailing_dims);
    let logical_strides = crate::helpers::shape_arith::row_major_strides(&logical_shape);

    // runtime_shape: row count for the leading axis, static for the rest.
    let mut runtime_shape = vec![runtime_row_count];
    for &td in trailing_dims {
        runtime_shape.push(ScalarValue::new(Some(td as i64), None));
    }
    let runtime_strides: Vec<ScalarValue<i64>> = logical_strides
        .iter()
        .map(|&s| ScalarValue::new(Some(s as i64), None))
        .collect();

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides,
            runtime_length,
            runtime_rank: ScalarValue::new(Some((1 + trailing_dims.len()) as i64), None),
            runtime_shape,
            runtime_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositeData;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,

            value_id: ValueId::next(),
        })
    }

    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let lst = list_of(leaves);
        crate::helpers::static_array::base::to_static_array(b, &lst).unwrap()
    }

    fn make_1d_bool(b: &mut IRBuilder, bits: &[bool]) -> Value {
        let leaves: Vec<Value> = bits.iter().map(|x| b.ir_constant_bool(*x)).collect();
        let lst = list_of(leaves);
        crate::helpers::static_array::base::to_static_array(b, &lst).unwrap()
    }

    #[test]
    fn boolean_construction_keeps_boolean_cells() {
        let mut b = IRBuilder::new();
        let m = make_1d_bool(&mut b, &[true, false, true]);
        match &m {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &m);
        for c in &cells {
            assert!(matches!(c, Value::Boolean(_)));
        }
        assert!(crate::helpers::array_ops::is_bool_array(&b, &m));
    }

    #[test]
    fn static_mask_read_returns_static_array() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[10, 20, 30, 40]);
        let mask = make_1d_bool(&mut b, &[true, false, true, false]);
        let out = try_apply_boolean_mask_read(&mut b, &arr, &mask).expect("native path");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2]),
            _ => panic!("expected static StaticArray output"),
        }
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(10), Some(30)]);
    }

    #[test]
    fn static_mask_write_scalar() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[10, 20, 30, 40]);
        let mask = make_1d_bool(&mut b, &[true, false, true, false]);
        let zero = b.ir_constant_int(0);
        let out = try_apply_boolean_mask_write(&mut b, &arr, &mask, &zero).expect("native path");
        // Cache invalidated; next read goes through storage_read_at. We
        // don't have an easy way to read post-write here without running
        // the proving backend, so just confirm we got back a StaticArray.
        match out {
            Value::StaticArray { .. } => {}
            _ => panic!("expected StaticArray return"),
        }
    }

    #[test]
    fn returns_none_for_non_boolean_mask() {
        let mut b = IRBuilder::new();
        let arr = make_1d_int(&mut b, &[1, 2, 3]);
        let other_int = make_1d_int(&mut b, &[0, 1, 1]);
        assert!(try_apply_boolean_mask_read(&mut b, &arr, &other_int).is_none());
    }

    /// 2-D arr with a 1-D leading-axis static mask → StaticArray of shape
    /// `(kept_rows, trailing)` with the kept rows concatenated in order.
    #[test]
    fn static_prefix_mask_read_returns_2d_static_array() {
        let mut b = IRBuilder::new();
        // arr shape (3, 2) values: [[10,11],[20,21],[30,31]].
        let arr_2d = {
            let leaves: Vec<Value> = [10, 11, 20, 21, 30, 31]
                .iter()
                .map(|n| b.ir_constant_int(*n))
                .collect();
            crate::helpers::static_array::base::build_static_array_from_flat(
                &mut b,
                leaves,
                vec![3, 2],
                NumberType::Integer,
            )
        };
        // mask shape (3,) → keep rows 0 and 2.
        let mask = make_1d_bool(&mut b, &[true, false, true]);
        let out = try_apply_boolean_mask_read(&mut b, &arr_2d, &mask).expect("native path");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2, 2]),
            _ => panic!("expected static StaticArray output"),
        }
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(10), Some(11), Some(30), Some(31)]);
    }

    /// Prefix mask (mask.shape strictly shorter than arr.shape) that
    /// happens to be fully static still emits a StaticArray (no fallback
    /// to the DynamicNDArray path).
    #[test]
    fn static_mask_with_zero_kept_rows_returns_2d_static_array() {
        let mut b = IRBuilder::new();
        let arr_2d = {
            let leaves: Vec<Value> = [1, 2, 3, 4, 5, 6]
                .iter()
                .map(|n| b.ir_constant_int(*n))
                .collect();
            crate::helpers::static_array::base::build_static_array_from_flat(
                &mut b,
                leaves,
                vec![3, 2],
                NumberType::Integer,
            )
        };
        let mask = make_1d_bool(&mut b, &[false, false, false]);
        let out = try_apply_boolean_mask_read(&mut b, &arr_2d, &mask).expect("native path");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![0, 2]),
            _ => panic!("expected static StaticArray output"),
        }
    }

    /// Mask shape that is neither equal to nor a prefix of arr.shape
    /// must still panic with the numpy-style message.
    #[test]
    #[should_panic(expected = "boolean index did not match indexed array")]
    fn prefix_mismatch_still_panics() {
        let mut b = IRBuilder::new();
        let arr_2d = {
            let leaves: Vec<Value> = (0..6).map(|n| b.ir_constant_int(n)).collect();
            crate::helpers::static_array::base::build_static_array_from_flat(
                &mut b,
                leaves,
                vec![3, 2],
                NumberType::Integer,
            )
        };
        // mask (2,) does NOT match arr_shape[..1] == [3].
        let mask = make_1d_bool(&mut b, &[true, false]);
        let _ = try_apply_boolean_mask_read(&mut b, &arr_2d, &mask);
    }
}
