//! Native read paths for `Value::StaticArray`.
//!
//! P2 of `compiler.epic-segment-native-static-arrays`: reads (element,
//! row/column view, multi-dim element, slice, for-loop iteration) operate
//! directly on the segment-backed representation without first materialising
//! the array via `to_value_list`. Static-index reads prefer the cached wire
//! held in `IRBuilder::static_array_payload` (free); dynamic-index reads emit
//! a single `ir_read_memory` op (O(1) constraint cost vs the legacy O(N) mux
//! chain in `dynamic_list_subscript`). View vs copy policy:
//!
//! - **Single-axis-0 indexing** producing a (D-1)-rank slice (`arr[i]` on a
//!   ≥2-D array): view — same segment, adjusted offset, dropped axis-0
//!   stride/shape. Cheap and matches NumPy semantics; the chained subscript
//!   form `arr[i][j]` lands here on the first step.
//! - **Static slice** (`arr[a:b]`, `arr[a:b, c]`):
//!   - positive step, including non-contiguous slabs: aliasing view (same
//!     segment, adjusted offset, shape and strides).
//!   - reversed slices: materialise while StaticArray strides remain unsigned.
//! - **Dynamic slice** (`arr[a:b]` with runtime `a` / `b`): always
//!   materialise (mirrors `dyn_slice_axis`).
//!
//! Cache policy: views produced by this module DO inherit the original cache
//! entry's segment_id — so reads via `static_array_payload.get(segment_id)`
//! still return the original wires. The `offset` field of the view shifts
//! the cache lookup window correctly.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::{scalar_i64_to_value, value_to_scalar_i64};
use crate::types::{NumberType, SliceIndex, Value, ValueId};

use super::super::shape_arith::{decode_coords, row_major_strides};

// ────────────────────────────────────────────────────────────────────────
// Element / row helpers
// ────────────────────────────────────────────────────────────────────────

/// Read a single leaf from a `Value::StaticArray` at the given flat (linear)
/// offset relative to `offset` (i.e. element `flat_idx` of the logical
/// array). Prefers the cached original wire; falls back to a single
/// `ir_read_memory` if no cache entry exists.
fn read_leaf_at_flat(
    b: &mut IRBuilder,
    segment_id: u32,
    base_offset: usize,
    flat_idx: usize,
    dtype: NumberType,
) -> Value {
    let abs = base_offset + flat_idx;
    if let Some(cached) = b.static_array_payload.get(&segment_id) {
        if abs < cached.len() {
            return cached[abs].clone();
        }
    }
    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }
    let addr = b.ir_constant_int(abs as i64);
    let raw = b.ir_read_memory(segment_id, &addr);
    scalar_i64_to_value(&value_to_scalar_i64(&raw), dtype)
}

/// Read a single leaf at a runtime address. Emits one `ir_read_memory` op
/// and no bounds check: the caller must have bounds-checked every index
/// the address is built from (`checked_axis_index`), or otherwise proved
/// that the address lies inside the array.
fn read_leaf_at_checked_address(
    b: &mut IRBuilder,
    segment_id: u32,
    addr: &Value,
    dtype: NumberType,
) -> Value {
    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }
    let raw = b.ir_read_memory(segment_id, addr);
    scalar_i64_to_value(&value_to_scalar_i64(&raw), dtype)
}

/// Read a runtime-addressed leaf whose original address is valid only under
/// `guard_condition`. When the guard is false, select a known-valid fallback
/// address before the memory read so the skipped witness is not hiding an
/// unconditional out-of-range access.
fn read_leaf_at_dynamic_with_guarded_safe_fallback_addr(
    b: &mut IRBuilder,
    segment_id: u32,
    addr: &Value,
    fallback_addr: i64,
    guard_condition: &Value,
    dtype: NumberType,
) -> Value {
    let fallback = b.ir_constant_int(fallback_addr);
    let safe_addr = b.ir_select_i(guard_condition, addr, &fallback);
    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }
    let raw = b.ir_read_memory(segment_id, &safe_addr);
    scalar_i64_to_value(&value_to_scalar_i64(&raw), dtype)
}

/// Build a runtime address from `offset + sum_k(idx[k] * strides[k])`.
fn compute_addr(
    b: &mut IRBuilder,
    base_offset: usize,
    strides: &[usize],
    indices: &[Value],
    shape: &[usize],
) -> Value {
    let mut static_sum: i64 = base_offset as i64;
    let mut dynamic_parts: Vec<Value> = Vec::new();
    for (ax, idx) in indices.iter().enumerate() {
        let stride = strides[ax] as i64;
        if let Some(i) = idx.int_val() {
            let i = if i < 0 { shape[ax] as i64 + i } else { i };
            static_sum += i * stride;
        } else {
            let stride_val = b.ir_constant_int(stride);
            dynamic_parts.push(b.ir_mul_i(idx, &stride_val));
        }
    }
    if dynamic_parts.is_empty() {
        b.ir_constant_int(static_sum)
    } else {
        let mut acc = b.ir_constant_int(static_sum);
        for p in &dynamic_parts {
            acc = b.ir_add_i(&acc, p);
        }
        acc
    }
}

const SMALL_DYNAMIC_SINGLE_READ_SELECT_LIMIT: usize = 1024;

fn select_typed(
    b: &mut IRBuilder,
    dtype: NumberType,
    cond: &Value,
    true_val: &Value,
    false_val: &Value,
) -> Value {
    match dtype {
        NumberType::Float => b.ir_select_f(cond, true_val, false_val),
        NumberType::Integer => b.ir_select_i(cond, true_val, false_val),
        NumberType::Complex => {
            panic!("select_typed does not handle Complex StaticArray reads")
        }
    }
}

use super::base::checked_axis_index;

/// One runtime index axis of a [`PayloadGather`]: the index, already
/// wrapped and bounds-checked on its axis (`checked_axis_index`), the
/// axis's stride in the segment and its length.
struct GatherAxis {
    index: Value,
    stride: usize,
    len: usize,
}

/// Runtime-index reads of a static array whose cells are known values
/// (`IRBuilder::static_array_payload`), as select chains instead of a
/// memory read.
///
/// `cells[base + sum_k i_k * stride_k]` is selected one index axis at a
/// time, outermost axis last: for the inner axes the selection is a
/// "plane" `cells[base' + ...]`, cached per (base', inner indices), so reads
/// that differ only in their outer indices share their inner selections
/// (a 2-D gather costs one chain per distinct row and column index instead
/// of one chain over the product of the two axes per read). Each chain
/// starts from coordinate 0 and selects coordinate `p` under `index == p`.
///
/// Soundness: every index is wrapped and bounds-checked on its axis before
/// it reaches the gather, so exactly one coordinate test holds per axis and
/// the chain yields the indexed cell; the equality tests are shared by
/// every chain over the same index.
struct PayloadGather {
    cells: Vec<Value>,
    dtype: NumberType,
    hits: std::collections::HashMap<(crate::types::StmtId, usize), Value>,
    planes: std::collections::HashMap<(i64, Vec<crate::types::StmtId>), Value>,
}

impl PayloadGather {
    /// The gather over `segment_id`'s known cells, or `None` when the
    /// cells are not known (a runtime write invalidated them) or the
    /// dtype is Complex.
    fn of_segment(b: &mut IRBuilder, segment_id: u32, dtype: NumberType) -> Option<Self> {
        if dtype == NumberType::Complex {
            return None;
        }
        if b.static_array_product_provenance.contains_key(&segment_id) {
            super::base::force_static_array_product_provenance(b, segment_id);
        }
        let cells = b.static_array_payload.get(&segment_id)?.clone();
        Some(Self {
            cells,
            dtype,
            hits: std::collections::HashMap::new(),
            planes: std::collections::HashMap::new(),
        })
    }

    /// Whether a read over `axes` stays within the select budget.
    fn affordable(axes: &[GatherAxis]) -> bool {
        let choices: usize = axes.iter().map(|a| a.len).product();
        !axes.is_empty() && choices > 0 && choices <= SMALL_DYNAMIC_SINGLE_READ_SELECT_LIMIT
    }

    fn hit(&mut self, b: &mut IRBuilder, index: &Value, coord: usize) -> Option<Value> {
        let key = (index.stmt_id()?, coord);
        if let Some(hit) = self.hits.get(&key) {
            return Some(hit.clone());
        }
        let coord_v = b.ir_constant_int(coord as i64);
        let hit = b.ir_equal_i(index, &coord_v);
        self.hits.insert(key, hit.clone());
        Some(hit)
    }

    /// `cells[base + sum_k axes[k].index * axes[k].stride]`, or `None` when
    /// a candidate lies outside the known cells or an index has no
    /// statement (the caller then falls back to a memory read).
    fn read(&mut self, b: &mut IRBuilder, base: i64, axes: &[GatherAxis]) -> Option<Value> {
        let Some((outer, inner)) = axes.split_first() else {
            return self.cells.get(usize::try_from(base).ok()?).cloned();
        };
        if inner.is_empty() {
            let (cells, base) = (&self.cells, usize::try_from(base).ok()?);
            if let Some(output) = sorted_output_at_index(b, cells, base, outer.stride, outer.len, &outer.index) {
                return Some(output);
            }
        }
        let key = (base, axes.iter().map(|a| a.index.stmt_id()).collect::<Option<Vec<_>>>()?);
        if let Some(plane) = self.planes.get(&key) {
            return Some(plane.clone());
        }
        let mut acc = self.read(b, base, inner)?;
        for coord in 1..outer.len {
            let candidate = self.read(b, base + (coord * outer.stride) as i64, inner)?;
            let hit = self.hit(b, &outer.index, coord)?;
            acc = select_typed(b, self.dtype, &hit, &candidate, &acc);
        }
        self.planes.insert(key, acc.clone());
        Some(acc)
    }
}

/// The read `cells[base + t * stride]` (`t` a runtime index, wrapped and
/// bounds-checked into `[0, len)` everywhere), when those `len` cells are
/// consecutive outputs of one sort (`np.sort(a)[t]`, `np.sort(a)[::-1][t]`,
/// `np.argsort(a)[1:][t]`): the sort's output at the runtime rank `r0 + t`
/// (or `r0 - t`). The sort lowering then pins that one output (or reads it
/// from the sorted outputs, whichever is cheaper) instead of every output
/// being kept for a selection. `None` for any other cells.
fn sorted_output_at_index(
    b: &mut IRBuilder,
    cells: &[Value],
    base: usize,
    stride: usize,
    len: usize,
    index: &Value,
) -> Option<Value> {
    let cell = |coord: usize| cells.get(base + coord * stride);
    // A sort lane at a constant rank: its operator instance, output, rank.
    let sort_lane = |b: &IRBuilder, v: &Value| {
        let p = b.array_op_placeholder_of(v).filter(|p| p.op.kind == crate::lower::OpKind::Sort)?;
        Some((p.shared.clone(), p.op.output, i64::from(p.op.position?)))
    };
    let (elements, output, r0) = sort_lane(b, cell(0)?)?;
    let step: i64 = match len {
        1 => 1,
        _ => sort_lane(b, cell(1)?)?.2 - r0,
    };
    if step != 1 && step != -1 {
        return None;
    }
    for coord in 0..len {
        let (other, other_output, rank) = sort_lane(b, cell(coord)?)?;
        let same_sort = std::sync::Arc::ptr_eq(&other, &elements);
        if !same_sort || other_output != output || rank != r0 + step * coord as i64 {
            return None;
        }
    }
    let r0 = b.ir_constant_int(r0);
    let rank = if step == 1 { b.ir_add_i(&r0, index) } else { b.ir_sub_i(&r0, index) };
    Some(b.ir_sorted_at_runtime_rank(&elements, &rank, output == crate::lower::sort::INDICES))
}

/// [`sorted_output_at_index`] over a 1-D view of `segment_id`'s known cells.
fn sorted_output_of_segment(
    b: &mut IRBuilder,
    segment_id: u32,
    offset: usize,
    stride: usize,
    len: usize,
    index: &Value,
) -> Option<Value> {
    let cells = b.static_array_payload.get(&segment_id)?.clone();
    sorted_output_at_index(b, &cells, offset, stride, len, index)
}

enum FancyAxisPlan {
    Single(Value),
    Range {
        coords: Vec<usize>,
    },
    Fancy {
        values: Vec<Value>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
}

fn integer_fancy_index_payload(
    b: &mut IRBuilder,
    value: &Value,
) -> Option<(Vec<Value>, Vec<usize>)> {
    let list_value = match value {
        Value::StaticArray { .. } => super::base::to_value_list(b, value),
        Value::List(_) | Value::Tuple(_) => value.clone(),
        _ => return None,
    };
    let shape = super::super::composite::get_composite_shape(&list_value);
    if shape.is_empty() {
        return None;
    }
    let flat = super::super::composite::flatten_composite_with_builder(b, &list_value);
    if flat.iter().all(|v| matches!(v, Value::Integer(_))) {
        Some((flat, shape))
    } else {
        None
    }
}

fn static_range_coords(
    shape: &[usize],
    axis: usize,
    start: &Option<Value>,
    stop: &Option<Value>,
    step: &Option<Value>,
) -> Option<Vec<usize>> {
    let len_i = shape[axis] as i64;
    let is_absent =
        |v: &Option<Value>| -> bool { matches!(v.as_ref(), Some(Value::None) | None) };
    let int_of = |v: &Option<Value>| -> Option<i64> {
        match v.as_ref() {
            Some(Value::None) | None => None,
            Some(val) => val.int_val(),
        }
    };
    let st_absent = is_absent(step);
    let s_absent = is_absent(start);
    let e_absent = is_absent(stop);
    let st_int = int_of(step);
    let s_int = int_of(start);
    let e_int = int_of(stop);
    if !(st_absent || st_int.is_some())
        || !(s_absent || s_int.is_some())
        || !(e_absent || e_int.is_some())
    {
        return None;
    }
    let st = st_int.unwrap_or(1);
    assert!(st != 0, "slice step cannot be zero");
    let s = if s_absent {
        if st > 0 {
            0
        } else {
            len_i - 1
        }
    } else {
        let raw = s_int.unwrap();
        if raw < 0 {
            (len_i + raw).max(0)
        } else {
            raw.min(len_i)
        }
    };
    let e = if e_absent {
        if st > 0 {
            len_i
        } else {
            -1
        }
    } else {
        let raw = e_int.unwrap();
        if raw < 0 {
            (len_i + raw).max(0)
        } else {
            raw.min(len_i)
        }
    };
    let mut coords = Vec::new();
    if st > 0 {
        let mut i = s;
        while i < e {
            coords.push(i as usize);
            i += st;
        }
    } else {
        let mut i = s;
        while i > e {
            coords.push(i as usize);
            i += st;
        }
    }
    Some(coords)
}

#[allow(clippy::too_many_arguments)]
/// NumPy advanced indexing on a StaticArray: any number of integer-array
/// index axes (broadcast together per numpy rules), mixed with static
/// Ranges and scalar Singles. Broadcast dims land at the position of the
/// first index array when the array axes are contiguous in the index
/// tuple, else first. Returns `None` for shapes this path doesn't cover
/// (Ellipsis/NewAxis, dynamic range bounds, no array axis at all).
fn try_fancy_subscript(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    imag_seg: Option<u32>,
    strides: &[usize],
    offset: usize,
    indices: &[SliceIndex],
) -> Option<Value> {
    let rank = shape.len();
    if indices.len() > rank {
        return None;
    }

    let mut plans: Vec<FancyAxisPlan> = Vec::with_capacity(rank);
    let mut fancy_positions: Vec<usize> = Vec::new();
    let mut fancy_shapes: Vec<Vec<usize>> = Vec::new();

    for ax in 0..rank {
        let idx = indices
            .get(ax)
            .cloned()
            .unwrap_or(SliceIndex::Range(None, None, None));
        match idx {
            SliceIndex::Single(v) => {
                if let Some((values, idx_shape)) = integer_fancy_index_payload(b, &v) {
                    fancy_positions.push(ax);
                    fancy_shapes.push(idx_shape.clone());
                    let idx_strides = row_major_strides(&idx_shape);
                    plans.push(FancyAxisPlan::Fancy {
                        values,
                        shape: idx_shape,
                        strides: idx_strides,
                    });
                } else {
                    plans.push(FancyAxisPlan::Single(v));
                }
            }
            SliceIndex::Range(start, stop, step) => {
                let coords = static_range_coords(shape, ax, &start, &stop, &step)?;
                plans.push(FancyAxisPlan::Range { coords });
            }
            SliceIndex::Ellipsis | SliceIndex::NewAxis => return None,
        }
    }

    if fancy_positions.is_empty() {
        return None;
    }

    // Broadcast all index-array shapes together (right-aligned).
    let bndim = fancy_shapes.iter().map(|sh| sh.len()).max().unwrap();
    let mut bshape = vec![1usize; bndim];
    for sh in &fancy_shapes {
        for (d, &n) in sh.iter().enumerate() {
            let bd = bndim - sh.len() + d;
            if n == 1 {
                continue;
            }
            if bshape[bd] == 1 {
                bshape[bd] = n;
            } else if bshape[bd] != n {
                panic!(
                    "shape mismatch: indexing arrays could not be broadcast together \
                     ({:?} vs {:?})",
                    bshape, sh
                );
            }
        }
    }

    // Placement of the broadcast block (numpy rule): at the first array
    // axis when the array axes are contiguous in the index tuple (scalar
    // Singles do not separate them), else first.
    let first_f = fancy_positions[0];
    let last_f = *fancy_positions.last().unwrap();
    let contiguous = !plans[first_f..=last_f]
        .iter()
        .any(|p| matches!(p, FancyAxisPlan::Range { .. }));

    let mut out_shape: Vec<usize> = Vec::new();
    let mut bstart = 0usize;
    let mut range_out_axis: Vec<Option<usize>> = vec![None; rank];
    if contiguous {
        for (ax, plan) in plans.iter().enumerate() {
            match plan {
                FancyAxisPlan::Single(_) => {}
                FancyAxisPlan::Range { coords } => {
                    range_out_axis[ax] = Some(out_shape.len());
                    out_shape.push(coords.len());
                }
                FancyAxisPlan::Fancy { .. } => {
                    if ax == first_f {
                        bstart = out_shape.len();
                        out_shape.extend(bshape.iter().copied());
                    }
                }
            }
        }
    } else {
        out_shape.extend(bshape.iter().copied());
        for (ax, plan) in plans.iter().enumerate() {
            if let FancyAxisPlan::Range { coords } = plan {
                range_out_axis[ax] = Some(out_shape.len());
                out_shape.push(coords.len());
            }
        }
    }

    // Wrap and bounds-check every index once per axis (numpy contract:
    // -n <= i < n). Doing it here — after the last early-return, before
    // the per-element loop — keeps the IR off abandoned paths and lets the
    // AssertU128Lt interval hook bound each index to [0, n), so the
    // backends' gather lowerings stay axis-sized instead of sweeping every
    // address the unbounded root could reach. After this loop every index
    // is non-negative and in range, so the reads below need no further
    // check.
    for (ax, plan) in plans.iter_mut().enumerate() {
        let axis_len = shape[ax];
        match plan {
            FancyAxisPlan::Fancy { values, .. } => {
                for val in values.iter_mut() {
                    *val = checked_axis_index(b, val, ax, axis_len, "fancy_index");
                }
            }
            FancyAxisPlan::Single(v) => {
                *v = checked_axis_index(b, v, ax, axis_len, "fancy_index");
            }
            FancyAxisPlan::Range { .. } => {}
        }
    }

    let total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);
    let mut leaves: Vec<Value> = Vec::with_capacity(total);

    for flat_out in 0..total {
        let out_coords = decode_coords(flat_out, &out_shape, &out_strides);
        let mut static_addr = offset as i64;
        let mut dynamic_parts: Vec<Value> = Vec::new();
        // The runtime indices of this read with their axis (for a gather by
        // an argsort lane, below).
        let mut dynamic_axes: Vec<(Value, usize)> = Vec::new();

        for (src_axis, plan) in plans.iter().enumerate() {
            match plan {
                FancyAxisPlan::Single(v) => {
                    if let Some(i) = v.int_val() {
                        // Already wrapped + bounds-checked above.
                        static_addr += i * strides[src_axis] as i64;
                    } else {
                        // Already wrapped + bounds-checked above.
                        let stride_val = b.ir_constant_int(strides[src_axis] as i64);
                        dynamic_parts.push(b.ir_mul_i(v, &stride_val));
                        dynamic_axes.push((v.clone(), src_axis));
                    }
                }
                FancyAxisPlan::Range { coords } => {
                    let oa = range_out_axis[src_axis].unwrap();
                    static_addr += coords[out_coords[oa]] as i64 * strides[src_axis] as i64;
                }
                FancyAxisPlan::Fancy {
                    values,
                    shape: idx_shape,
                    strides: idx_strides,
                } => {
                    // Map the broadcast coordinates onto this index array
                    // (right-aligned; size-1 dims pin to coordinate 0).
                    let mut idx_flat = 0usize;
                    for d in 0..idx_shape.len() {
                        let coord = if idx_shape[d] == 1 {
                            0
                        } else {
                            out_coords[bstart + bndim - idx_shape.len() + d]
                        };
                        idx_flat += coord * idx_strides[d];
                    }
                    let idx_val = values[idx_flat].clone();
                    if let Some(i) = idx_val.int_val() {
                        // Already wrapped + bounds-checked above.
                        static_addr += i * strides[src_axis] as i64;
                    } else {
                        // Already wrapped + bounds-checked above.
                        let stride_val = b.ir_constant_int(strides[src_axis] as i64);
                        dynamic_parts.push(b.ir_mul_i(&idx_val, &stride_val));
                        dynamic_axes.push((idx_val, src_axis));
                    }
                }
            }
        }

        // One runtime index that is an argsort lane: the gathered line is a
        // payload carried by that sort (`lower::sort::payload`).
        if let ([(index, axis)], false) = (&dynamic_axes[..], dtype == NumberType::Complex) {
            let line: Option<Vec<Value>> = b.static_array_payload.get(&segment_id).and_then(|cells| {
                (0..shape[*axis])
                    .map(|j| cells.get(static_addr as usize + j * strides[*axis]).cloned())
                    .collect()
            });
            if let Some(gathered) = line.and_then(|line| crate::lower::sort::payload_at_argsort_lane(b, index, &line)) {
                leaves.push(gathered);
                continue;
            }
        }

        if dynamic_parts.is_empty() {
            let abs = static_addr as usize;
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf(b, segment_id, im, abs));
            } else {
                leaves.push(read_leaf_at_flat(b, segment_id, 0, abs, dtype));
            }
        } else {
            let mut addr = b.ir_constant_int(static_addr);
            for part in &dynamic_parts {
                addr = b.ir_add_i(&addr, part);
            }
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf_dynamic(b, segment_id, im, &addr));
            } else {
                leaves.push(read_leaf_at_checked_address(b, segment_id, &addr, dtype));
            }
        }
    }

    if dtype == NumberType::Complex {
        let mut reals = Vec::with_capacity(leaves.len());
        let mut imags = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            match leaf {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!(),
            }
        }
        return Some(super::base::build_static_array_from_flat_complex(
            b, reals, imags, out_shape,
        ));
    }

    Some(super::base::build_static_array_from_flat(
        b, leaves, out_shape, dtype,
    ))
}

/// Basic indexing with compile-time scalar indices and nonnegative strides
/// returns a NumPy view. This includes strided columns and mixed scalar/slice
/// indexing; copying these loses mutations through aliases and ufunc out=.
fn static_basic_view(b: &mut IRBuilder, value: &Value, indices: &[SliceIndex]) -> Option<Value> {
    let Value::StaticArray { dtype, shape, segment_id, strides, offset,
        imag_segment_id, .. } = value else { return None };
    if indices.len() > shape.len() { return None; }
    let bound = |v: &Option<Value>| -> Option<Option<i64>> {
        match v { None | Some(Value::None) => Some(None), Some(v) => v.int_val().map(Some) }
    };
    let mut out_shape = Vec::new();
    let mut out_strides = Vec::new();
    let mut out_offset = *offset;
    for ax in 0..shape.len() {
        match indices.get(ax) {
            None => {
                out_shape.push(shape[ax]);
                out_strides.push(strides[ax]);
            }
            Some(SliceIndex::Single(value)) => {
                let raw = value.int_val()?;
                let index = if raw < 0 { raw.saturating_add(shape[ax] as i64) } else { raw };
                if !(index >= 0 && index < shape[ax] as i64) {
                    // Out of range: the checked read path reports it (a
                    // compile error, or an assertion that the code does not
                    // run when it runs only under a runtime condition).
                    return None;
                }
                out_offset += index as usize * strides[ax];
            }
            Some(SliceIndex::Range(start, stop, step)) => {
                let start = bound(start)?;
                let stop = bound(stop)?;
                let step = bound(step)?.unwrap_or(1);
                assert!(step != 0, "slice step cannot be zero");
                // StaticArray strides are unsigned. Keep reversed slicing
                // on the existing materializing path until signed strides
                // can be represented throughout the array implementation.
                if step < 0 { return None; }
                let len = shape[ax] as i64;
                let normalize = |raw: i64| {
                    let normalized = if raw < 0 { raw.saturating_add(len) } else { raw };
                    normalized.max(0).min(len) as usize
                };
                let start = start.map(normalize).unwrap_or(0);
                let stop = stop.map(normalize).unwrap_or(shape[ax]);
                let step = step as usize;
                let length = if start >= stop { 0 } else { 1 + (stop - start - 1) / step };
                out_offset += start * strides[ax];
                out_shape.push(length);
                out_strides.push(strides[ax].checked_mul(step).expect("slice stride overflow"));
            }
            _ => return None,
        }
    }
    if out_shape.is_empty() { return None; }
    let out = Value::StaticArray { dtype: *dtype, shape: out_shape,
        segment_id: *segment_id, strides: out_strides, offset: out_offset,
        imag_segment_id: *imag_segment_id, value_id: ValueId::next() };
    if let (Some(input), Some(output)) = (value.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, input, output);
    }
    Some(out)
}

// ────────────────────────────────────────────────────────────────────────
// Public dispatch
// ────────────────────────────────────────────────────────────────────────

/// Subscript a `Value::StaticArray` natively (no materialisation to List).
/// Mirrors the responsibilities of `dyn_subscript` for `DynamicNDArray`, but
/// stays on the StaticArray representation when it can.
pub fn static_array_subscript(b: &mut IRBuilder, val: &Value, indices: &[SliceIndex]) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
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
        _ => panic!("static_array_subscript: expected Value::StaticArray"),
    };

    if indices.is_empty() {
        return val.clone();
    }

    match tuple_index(b, indices, &shape) {
        TupleIndex::NotATuple => {}
        TupleIndex::Mask(mask) => {
            return super::mask::try_apply_boolean_mask_read(b, val, &mask)
                .expect("nonzero mask has the array's shape")
        }
        TupleIndex::Axes(expanded) => return static_array_subscript(b, val, &expanded),
    }
    crate::helpers::array_ops::reject_unsupported_index_arrays(b, indices);

    // Expand a single Ellipsis (`...`) into the right number of full-range
    // slices, based on the source rank.
    if indices.iter().any(|i| matches!(i, SliceIndex::Ellipsis)) {
        let consumed: usize = indices
            .iter()
            .filter(|i| matches!(i, SliceIndex::Single(_) | SliceIndex::Range(_, _, _)))
            .count();
        let num_colons = shape.len().saturating_sub(consumed);
        let mut expanded: Vec<SliceIndex> = Vec::with_capacity(indices.len() - 1 + num_colons);
        let mut seen = false;
        for idx in indices {
            match idx {
                SliceIndex::Ellipsis if !seen => {
                    seen = true;
                    for _ in 0..num_colons {
                        expanded.push(SliceIndex::Range(None, None, None));
                    }
                }
                SliceIndex::Ellipsis => panic!("an index can only have a single ellipsis ('...')"),
                other => expanded.push(other.clone()),
            }
        }
        return static_array_subscript(b, val, &expanded);
    }

    // NewAxis is metadata-only when the rest of the subscript stays on the
    // StaticArray path. Keep it here so tensor expressions such as
    // `arr[..., np.newaxis]` can feed StaticArray elementwise provenance.
    if indices.iter().any(|i| matches!(i, SliceIndex::NewAxis)) {
        let mut cleaned: Vec<SliceIndex> = Vec::with_capacity(indices.len());
        let mut insert_positions: Vec<usize> = Vec::new();
        let mut out_axis = 0usize;
        for idx in indices {
            match idx {
                SliceIndex::NewAxis => {
                    insert_positions.push(out_axis);
                    out_axis += 1;
                }
                SliceIndex::Single(_) => cleaned.push(idx.clone()),
                SliceIndex::Range(_, _, _) => {
                    cleaned.push(idx.clone());
                    out_axis += 1;
                }
                SliceIndex::Ellipsis => unreachable!("ellipsis expanded before NewAxis handling"),
            }
        }
        let base = if cleaned.is_empty() {
            val.clone()
        } else {
            static_array_subscript(b, val, &cleaned)
        };
        if let Value::StaticArray {
            dtype,
            mut shape,
            segment_id,
            mut strides,
            offset,
            imag_segment_id,
            ..
        } = base
        {
            for pos in insert_positions {
                let pos = pos.min(shape.len());
                shape.insert(pos, 1);
                strides.insert(pos, 0);
            }
            return Value::StaticArray {
                dtype,
                shape,
                segment_id,
                strides,
                offset,
                imag_segment_id,
                value_id: ValueId::next(),
            };
        }
        let lst = super::base::to_value_list(b, &base);
        let data = match &lst {
            Value::List(d) => d.clone(),
            _ => return lst,
        };
        return super::super::ndarray::multidim_subscript(b, &data, indices);
    }

    if let Some(out) = try_fancy_subscript(
        b, dtype, &shape, segment_id, imag_seg, &strides, offset, indices,
    ) {
        return out;
    }

    if let Some(view) = static_basic_view(b, val, indices) {
        return view;
    }

    // ── Fast path: all-Single, fully-resolved index across all axes →
    //    leaf read (static or dynamic).
    let all_single = indices.iter().all(|s| matches!(s, SliceIndex::Single(_)));
    if all_single && indices.len() == shape.len() {
        // Wrap and bounds-check every axis index (compile-time for
        // literals); afterwards each index is in `[0, shape[ax])`, so the
        // address below lies inside the array.
        let idx_vals: Vec<Value> = indices
            .iter()
            .enumerate()
            .map(|(ax, s)| match s {
                SliceIndex::Single(v) => {
                    checked_axis_index(b, v, ax, shape[ax], "static_array_read")
                }
                _ => unreachable!(),
            })
            .collect();
        let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
        if all_static {
            let flat: usize = idx_vals
                .iter()
                .zip(&strides)
                .map(|(v, stride)| v.int_val().unwrap() as usize * stride)
                .sum();
            if dtype == NumberType::Complex {
                return read_complex_leaf(
                    b,
                    segment_id,
                    imag_seg.expect("Complex StaticArray missing imag_segment_id"),
                    offset + flat,
                );
            }
            return read_leaf_at_flat(b, segment_id, offset, flat, dtype);
        }
        if shape.len() == 1 {
            if let Some(out) = sorted_output_of_segment(b, segment_id, offset, strides[0], shape[0], &idx_vals[0]) {
                return out;
            }
        }
        let addr = compute_addr(b, offset, &strides, &idx_vals, &shape);
        if dtype == NumberType::Complex {
            return read_complex_leaf_dynamic(
                b,
                segment_id,
                imag_seg.expect("Complex StaticArray missing imag_segment_id"),
                &addr,
            );
        }
        return read_leaf_at_checked_address(b, segment_id, &addr, dtype);
    }

    // ── Single-index, ndim==1 → leaf or panic
    if indices.len() == 1 && shape.len() == 1 {
        match &indices[0] {
            SliceIndex::Single(v) => {
                // NumPy allows `arr[-1]`-style negative indices (one wrap,
                // not modular); `checked_axis_index` wraps with a cheap
                // `select(i < 0, i + len, i)` (skipped when the facts show
                // `i >= 0`, e.g. for an `argmax` result) and range-checks
                // the wrapped value.
                let i = checked_axis_index(b, v, 0, shape[0], "static_array_read");
                if let Some(i) = i.int_val() {
                    let flat = i as usize * strides[0];
                    if dtype == NumberType::Complex {
                        return read_complex_leaf(
                            b,
                            segment_id,
                            imag_seg.expect("Complex StaticArray missing imag_segment_id"),
                            offset + flat,
                        );
                    }
                    return read_leaf_at_flat(b, segment_id, offset, flat, dtype);
                }
                if let Some(out) = sorted_output_of_segment(b, segment_id, offset, strides[0], shape[0], &i) {
                    return out;
                }
                let off_val = b.ir_constant_int(offset as i64);
                let stride_val = b.ir_constant_int(strides[0] as i64);
                let scaled = b.ir_mul_i(&i, &stride_val);
                let addr = b.ir_add_i(&off_val, &scaled);
                if dtype == NumberType::Complex {
                    return read_complex_leaf_dynamic(
                        b,
                        segment_id,
                        imag_seg.expect("Complex StaticArray missing imag_segment_id"),
                        &addr,
                    );
                }
                return read_leaf_at_checked_address(b, segment_id, &addr, dtype);
            }
            SliceIndex::Range(s, e, st) => {
                return slice_axis_static_or_dynamic_1d(
                    b,
                    val,
                    dtype,
                    shape,
                    segment_id,
                    strides,
                    offset,
                    s.as_ref(),
                    e.as_ref(),
                    st.as_ref(),
                );
            }
            _ => unreachable!("Ellipsis/NewAxis already handled"),
        }
    }

    // ── Single-index on ndim>=2: row select on axis 0 → return a view
    //    (no materialisation). The chained `arr[i][j]` form lands here.
    if indices.len() == 1 && shape.len() >= 2 {
        match &indices[0] {
            SliceIndex::Single(v) => {
                let new_shape = shape[1..].to_vec();
                let new_strides = strides[1..].to_vec();
                let i = checked_axis_index(b, v, 0, shape[0], "static_array_read");
                if let Some(i) = i.int_val() {
                    let new_offset = offset + (i as usize) * strides[0];
                    return Value::StaticArray {
                        dtype,
                        shape: new_shape,
                        segment_id,
                        strides: new_strides,
                        offset: new_offset,
                        imag_segment_id: imag_seg,
                        value_id: ValueId::next(),
                    };
                }
                // Dynamic axis-0 index: materialise the row into a fresh
                // contiguous segment. We can't represent a dynamic-offset
                // view in the StaticArray variant.
                return materialise_axis0_dynamic_row(
                    b, dtype, &shape, segment_id, imag_seg, &strides, offset, &i,
                );
            }
            SliceIndex::Range(s, e, st) => {
                return slice_axis(
                    b,
                    val,
                    dtype,
                    &shape,
                    segment_id,
                    &strides,
                    offset,
                    0,
                    s.as_ref(),
                    e.as_ref(),
                    st.as_ref(),
                );
            }
            _ => unreachable!(),
        }
    }

    // ── Multi-index, mixed Single / Range across multiple axes.
    //    Strategy: if all Range bounds are static and all Single ints are
    //    static → return a contiguous view when possible, else materialise.
    //    Otherwise route through the general multidim materialiser.
    multidim_subscript_static_array(b, val, indices)
}

pub(crate) fn static_array_subscript_with_preproved_index_bounds(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
) -> Value {
    let (dtype, shape, segment_id, strides, offset, _imag_seg) = match val {
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
        _ => panic!(
            "static_array_subscript_with_preproved_index_bounds: expected Value::StaticArray"
        ),
    };

    if indices.len() == 1 && shape.len() == 1 {
        if let SliceIndex::Single(v) = &indices[0] {
            if v.int_val().is_none() {
                if dtype == NumberType::Complex {
                    return static_array_subscript(b, val, indices);
                }
                let off_val = b.ir_constant_int(offset as i64);
                let stride_val = b.ir_constant_int(strides[0] as i64);
                let scaled = b.ir_mul_i(v, &stride_val);
                let addr = b.ir_add_i(&off_val, &scaled);
                return read_leaf_at_checked_address(
                    b, segment_id, &addr, dtype,
                );
            }
        }
    }

    static_array_subscript(b, val, indices)
}

pub(crate) fn static_array_subscript_with_path_conditioned_preproved_index_bounds(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
    guard_condition: &Value,
) -> Value {
    let (dtype, shape, segment_id, strides, offset, _imag_seg) = match val {
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
        _ => panic!(
            "static_array_subscript_with_path_conditioned_preproved_index_bounds: expected Value::StaticArray"
        ),
    };

    if indices.len() == 1 && shape.len() == 1 {
        if let SliceIndex::Single(v) = &indices[0] {
            if v.int_val().is_none() {
                if dtype == NumberType::Complex {
                    return static_array_subscript(b, val, indices);
                }
                let off_val = b.ir_constant_int(offset as i64);
                let stride_val = b.ir_constant_int(strides[0] as i64);
                let scaled = b.ir_mul_i(v, &stride_val);
                let addr = b.ir_add_i(&off_val, &scaled);
                return read_leaf_at_dynamic_with_guarded_safe_fallback_addr(
                    b,
                    segment_id,
                    &addr,
                    offset as i64,
                    guard_condition,
                    dtype,
                );
            }
        }
    }

    static_array_subscript(b, val, indices)
}

// ────────────────────────────────────────────────────────────────────────
// Slice helpers
// ────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn slice_axis_static_or_dynamic_1d(
    b: &mut IRBuilder,
    src: &Value,
    dtype: NumberType,
    shape: Vec<usize>,
    segment_id: u32,
    strides: Vec<usize>,
    offset: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let len = shape[0];
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| {
            if matches!(val, Value::None) {
                None
            } else {
                val.int_val()
            }
        })
    };
    let is_present =
        |v: Option<&Value>| -> bool { matches!(v, Some(val) if !matches!(val, Value::None)) };
    let s_static = static_val(start);
    let e_static = static_val(stop);
    let st_static = static_val(step);

    let all_static = (!is_present(start) || s_static.is_some())
        && (!is_present(stop) || e_static.is_some())
        && (!is_present(step) || st_static.is_some());

    let out = if !all_static {
        // Dynamic bounds → materialise.
        materialise_dynamic_1d_slice(
            b, dtype, len, segment_id, strides[0], offset, start, stop, step,
        )
    } else {
        let len_i = len as i64;
        let st = st_static.unwrap_or(1);
        assert!(st != 0, "slice step cannot be zero");
        // Pythonic slice-bound defaults depend on step sign:
        //   step > 0 → start=0, stop=len
        //   step < 0 → start=len-1, stop=-1 (i.e. "before index 0")
        // For explicit bounds we keep the original Python clamping
        // (start clamps to [0, len], stop clamps to [0, len]).
        let s = if is_present(start) {
            let raw = s_static.unwrap();
            if raw < 0 {
                (len_i + raw).max(0)
            } else {
                raw.min(len_i)
            }
        } else if st > 0 {
            0
        } else {
            len_i - 1
        };
        let e = if is_present(stop) {
            let raw = e_static.unwrap();
            if raw < 0 {
                (len_i + raw).max(0)
            } else {
                raw.min(len_i)
            }
        } else if st > 0 {
            len_i
        } else {
            -1
        };

        if st == 1 && strides[0] == 1 {
            // Contiguous step=1 slab → view.
            let out_len = (e - s).max(0) as usize;
            let new_offset = offset + s as usize;
            // Preserve dual-segment imag for Complex views.
            let imag_seg = if let Value::StaticArray {
                imag_segment_id, ..
            } = src
            {
                *imag_segment_id
            } else {
                None
            };
            Value::StaticArray {
                dtype,
                shape: vec![out_len],
                segment_id,
                strides: vec![strides[0]],
                offset: new_offset,
                imag_segment_id: imag_seg,
                value_id: ValueId::next(),
            }
        } else {
            // Non-contiguous: materialise.
            let mut indices: Vec<i64> = Vec::new();
            if st > 0 {
                let mut i = s;
                while i < e {
                    indices.push(i);
                    i += st;
                }
            } else {
                let mut i = s;
                while i > e {
                    indices.push(i);
                    i += st;
                }
            }
            let out_len = indices.len();
            if dtype == NumberType::Complex {
                let imag_seg = if let Value::StaticArray {
                    imag_segment_id, ..
                } = src
                {
                    imag_segment_id.expect("Complex StaticArray missing imag_segment_id")
                } else {
                    unreachable!()
                };
                let mut reals: Vec<Value> = Vec::with_capacity(out_len);
                let mut imags: Vec<Value> = Vec::with_capacity(out_len);
                for src_i in &indices {
                    let flat = (*src_i as usize) * strides[0];
                    let abs = offset + flat;
                    let leaf = read_complex_leaf(b, segment_id, imag_seg, abs);
                    if let Value::Complex { real, imag } = leaf {
                        reals.push(Value::Float(real));
                        imags.push(Value::Float(imag));
                    }
                }
                super::base::build_static_array_from_flat_complex(b, reals, imags, vec![out_len])
            } else {
                let mut leaves: Vec<Value> = Vec::with_capacity(out_len);
                for src_i in &indices {
                    let flat = (*src_i as usize) * strides[0];
                    leaves.push(read_leaf_at_flat(b, segment_id, offset, flat, dtype));
                }
                super::base::build_static_array_from_flat(b, leaves, vec![out_len], dtype)
            }
        }
    };
    if let (Some(in_vid), Some(out_vid)) = (src.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `(start, stop, step, step sign)` of a slice with runtime bounds over an
/// axis of `len` elements, by Python's rules (see
/// [`crate::helpers::array_ops::python_slice_bounds`]).
fn python_bounds(
    b: &mut IRBuilder,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
    len: usize,
) -> (Value, Value, Value, Option<bool>) {
    let len_value = b.ir_constant_int(len as i64);
    let slice = crate::helpers::array_ops::python_slice_bounds(b, start, stop, step, &len_value, Some(len));
    (slice.start, slice.stop, slice.step, slice.step_positive)
}

#[allow(clippy::too_many_arguments)]
fn materialise_dynamic_1d_slice(
    b: &mut IRBuilder,
    dtype: NumberType,
    len: usize,
    segment_id: u32,
    axis_stride: usize,
    offset: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let max_out_len = len;

    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }

    // Python's slice semantics: any integer bound is valid; negative bounds
    // count from the end and every bound is clamped into the axis.
    let (start_ir, stop_ir, step_ir, step_positive) = python_bounds(b, start, stop, step, len);

    // See materialise_dynamic_axis_slice for the rationale on these elisions.
    let step_pos_known: bool = match step_positive {
        Some(positive) => positive,
        None => provable_strictly_positive(b, &step_ir),
    };
    // A normalized start is in [0, len] for a positive step.
    let start_nonneg: bool = match start_ir.int_val() {
        Some(s) => s >= 0,
        None => step_pos_known || provable_nonneg(b, &start_ir),
    };
    let step_is_one = step_ir.int_val() == Some(1);
    let fast = step_pos_known && start_nonneg;

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype);
    let zero = b.ir_constant_int(0);
    let len_val = b.ir_constant_int(len as i64);
    let stride_val = b.ir_constant_int(axis_stride as i64);
    let offset_val = b.ir_constant_int(offset as i64);

    let mut leaves: Vec<Value> = Vec::with_capacity(max_out_len);
    for i in 0..max_out_len {
        let i_const = b.ir_constant_int(i as i64);
        let src_idx = if step_is_one {
            b.ir_add_i(&start_ir, &i_const)
        } else {
            let off2 = b.ir_mul_i(&i_const, &step_ir);
            b.ir_add_i(&start_ir, &off2)
        };

        let in_bounds = if fast {
            // Note: NOT using `ir_less_than_i_bounded` here even though the
            // operands are non-negative. The bounded form expands to an
            // `(x as u128) < (y as u128)` cast which instantiates UltraHonk's
            // 128-bit range-check lookup tables — ~2700 gates fixed cost.
            // That cost amortizes only across ≤2-3 comparisons; for a
            // per-cell slice mask (one comparison per output cell, often
            // 8-100+ cells), the qlt sign-aware path (~30 gates each, no
            // lookup table) is strictly cheaper. The 2D `materialise_dynamic
            // _axis_slice` uses the bounded form only because there it was
            // already instantiated for other slice-bound discharges in the
            // same circuit; check both before flipping.
            let lt_len = b.ir_less_than_i(&src_idx, &len_val);
            let lt_stop = b.ir_less_than_i(&src_idx, &stop_ir);
            b.ir_logical_and(&lt_len, &lt_stop)
        } else {
            let ge_zero = b.ir_greater_than_or_equal_i(&src_idx, &zero);
            let lt_len = b.ir_less_than_i(&src_idx, &len_val);
            let in_range = b.ir_logical_and(&ge_zero, &lt_len);
            let lt_stop = b.ir_less_than_i(&src_idx, &stop_ir);
            let stop_bool = if step_pos_known {
                lt_stop
            } else {
                let gt_stop = b.ir_greater_than_i(&src_idx, &stop_ir);
                let step_pos = b.ir_greater_than_i(&step_ir, &zero);
                let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
                b.ir_bool_cast(&stop_ok)
            };
            b.ir_logical_and(&in_range, &stop_bool)
        };

        let max_idx = b.ir_constant_int(len as i64 - 1);
        // Keep qlt/qgt here for the same reason as in `in_bounds` above —
        // the bounded form's u128 lookup table is more expensive per-cell
        // than qlt when the slice mask runs over many cells.
        let is_over = b.ir_greater_than_i(&src_idx, &max_idx);
        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_idx);
        let clamped = if fast {
            clamped_hi
        } else {
            let is_neg = b.ir_less_than_i(&src_idx, &zero);
            b.ir_select_i(&is_neg, &zero, &clamped_hi)
        };

        let addr = if axis_stride == 1 {
            b.ir_add_i(&offset_val, &clamped)
        } else {
            let scaled = b.ir_mul_i(&clamped, &stride_val);
            b.ir_add_i(&offset_val, &scaled)
        };
        let elem = b.ir_read_memory(segment_id, &addr);
        let masked = if dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&masked), dtype));
    }

    // Compute the slice's actual runtime length so downstream ops (reshape,
    // aggregation) can verify shape constraints against the true length
    // rather than the over-allocated upper bound. Without this, a reshape on
    // a dynamic-bound slice would assert against `max_out_len` and silently
    // pass under field-arithmetic semantics (the root cause of the halo2
    // soundness hole on 0303-style kernels).
    let step_pos_2 = b.ir_greater_than_i(&step_ir, &zero);
    let stop_minus_start = b.ir_sub_i(&stop_ir, &start_ir);
    let start_minus_stop = b.ir_sub_i(&start_ir, &stop_ir);
    let diff = b.ir_select_i(&step_pos_2, &stop_minus_start, &start_minus_stop);
    let neg_step = b.ir_sub_i(&zero, &step_ir);
    let abs_step = b.ir_select_i(&step_pos_2, &step_ir, &neg_step);
    let diff_pos = b.ir_greater_than_i(&diff, &zero);
    let one = b.ir_constant_int(1);
    let abs_step_m1 = b.ir_sub_i(&abs_step, &one);
    let numerator = b.ir_add_i(&diff, &abs_step_m1);
    let divided = b.ir_floor_div_i(&numerator, &abs_step);
    let runtime_length = b.ir_select_i(&diff_pos, &divided, &zero);
    let runtime_len_sv = value_to_scalar_i64(&runtime_length);

    let out_segment = crate::helpers::segment::alloc_and_write_values(b, &leaves, dtype);
    // Slice provenance: valid output element i reads
    // base[offset + (start + i*step) * stride]. A downstream slice composes
    // this map instead of re-materialising (see IRBuilder::slice_views).
    if let Some(step_const) = step_ir.int_val() {
        if dtype != NumberType::Complex {
            b.provenance.set_slice_view(
                out_segment,
                crate::builder::SliceView {
                    base_segment: segment_id,
                    axis_len: len,
                    stride: axis_stride,
                    base_offset: offset,
                    start: start_ir.clone(),
                    step: step_const,
                },
            );
        }
    }
    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(
            &mut b.dim_table,
            0,
            max_out_len,
        )],
        max_out_len,
    );

    Value::DynamicNDArray(crate::types::DynamicNDArrayData {
        envelope,
        dtype,
        segment_id: out_segment,
        meta: crate::types::DynArrayMeta {
            logical_shape: vec![max_out_len],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: runtime_len_sv.clone(),
            runtime_rank: crate::types::ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_len_sv],
            runtime_strides: vec![crate::types::ScalarValue::new(Some(1), None)],
            runtime_offset: crate::types::ScalarValue::new(Some(0), None),
        },
        value_id: crate::types::ValueId::next(),
    })
}

#[allow(clippy::too_many_arguments)]
fn slice_axis(
    b: &mut IRBuilder,
    src: &Value,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    axis: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    if shape.len() == 1 {
        // The inner 1-D helper already emits the content-relay on return,
        // so don't double-relay here.
        return slice_axis_static_or_dynamic_1d(
            b,
            src,
            dtype,
            shape.to_vec(),
            segment_id,
            strides.to_vec(),
            offset,
            start,
            stop,
            step,
        );
    }
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| {
            if matches!(val, Value::None) {
                None
            } else {
                val.int_val()
            }
        })
    };
    let is_present =
        |v: Option<&Value>| -> bool { matches!(v, Some(val) if !matches!(val, Value::None)) };
    let s_static = static_val(start);
    let e_static = static_val(stop);
    let st_static = static_val(step);
    let all_static = (!is_present(start) || s_static.is_some())
        && (!is_present(stop) || e_static.is_some())
        && (!is_present(step) || st_static.is_some());

    let out = if !all_static {
        let padded = materialise_dynamic_axis_slice(
            b, dtype, shape, segment_id, strides, offset, axis, start, stop, step,
        );
        return runtime_shaped_slice(b, padded);
    } else {
        let len_i = shape[axis] as i64;
        let st = st_static.unwrap_or(1);
        assert!(st != 0, "slice step cannot be zero");
        // Pythonic slice-bound defaults depend on step sign — see
        // slice_axis_static_or_dynamic_1d for the parallel logic.
        let s = if is_present(start) {
            let raw = s_static.unwrap();
            if raw < 0 {
                (len_i + raw).max(0)
            } else {
                raw.min(len_i)
            }
        } else if st > 0 {
            0
        } else {
            len_i - 1
        };
        let e = if is_present(stop) {
            let raw = e_static.unwrap();
            if raw < 0 {
                (len_i + raw).max(0)
            } else {
                raw.min(len_i)
            }
        } else if st > 0 {
            len_i
        } else {
            -1
        };

        if st == 1 && axis == 0 {
            // Slice along axis 0 with step 1: contiguous view.
            let out_len = (e - s).max(0) as usize;
            let mut new_shape = shape.to_vec();
            new_shape[0] = out_len;
            let new_offset = offset + (s as usize) * strides[0];
            let imag_seg = if let Value::StaticArray {
                imag_segment_id, ..
            } = src
            {
                *imag_segment_id
            } else {
                None
            };
            Value::StaticArray {
                dtype,
                shape: new_shape,
                segment_id,
                strides: strides.to_vec(),
                offset: new_offset,
                imag_segment_id: imag_seg,
                value_id: ValueId::next(),
            }
        } else {
            // Non-contiguous along this axis OR slicing inner axis: materialise.
            let mut indices: Vec<i64> = Vec::new();
            if st > 0 {
                let mut i = s;
                while i < e {
                    indices.push(i);
                    i += st;
                }
            } else {
                let mut i = s;
                while i > e {
                    indices.push(i);
                    i += st;
                }
            }
            let out_axis_len = indices.len();
            let mut new_shape = shape.to_vec();
            new_shape[axis] = out_axis_len;
            let total: usize = new_shape.iter().product();
            let new_strides = row_major_strides(&new_shape);

            let mut leaves: Vec<Value> = Vec::with_capacity(total);
            for flat_out in 0..total {
                let coords = decode_coords(flat_out, &new_shape, &new_strides);
                let mut src_flat: usize = 0;
                for ax in 0..shape.len() {
                    let src_coord = if ax == axis {
                        indices[coords[ax]] as usize
                    } else {
                        coords[ax]
                    };
                    src_flat += src_coord * strides[ax];
                }
                leaves.push(read_leaf_at_flat(b, segment_id, offset, src_flat, dtype));
            }
            super::base::build_static_array_from_flat(b, leaves, new_shape, dtype)
        }
    };
    if let (Some(in_vid), Some(out_vid)) = (src.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// How a subscript whose single index is a tuple value is read.
pub(crate) enum TupleIndex {
    /// The index is not a single tuple value.
    NotATuple,
    /// The tuple came from `np.nonzero(m)` / `np.where(m)` on an array of
    /// this shape: indexing with it is indexing with the mask `m`.
    Mask(Value),
    /// Any other tuple `t`: NumPy reads `a[t]` as `a[t[0], t[1], ...]`, one
    /// entry per axis (a list, by contrast, is a fancy index on axis 0).
    Axes(Vec<SliceIndex>),
}

/// Classify a subscript `a[t]` whose index `t` is a tuple value (e.g. the
/// result of `np.unravel_index`). Both the read and the write path use it;
/// it used to be read as a fancy index on axis 0, so `a[(1, 0)]` gave rows
/// 1 and 0 and `a[np.unravel_index(k, a.shape)] = v` wrote whole rows.
pub(crate) fn tuple_index(b: &IRBuilder, indices: &[SliceIndex], shape: &[usize]) -> TupleIndex {
    let [SliceIndex::Single(Value::Tuple(tuple))] = indices else {
        return TupleIndex::NotATuple;
    };
    if let Some(mask) = b.nonzero_index_masks.get(&tuple.value_id) {
        if matches!(mask, Value::StaticArray { shape: m, .. } if m.as_slice() == shape) {
            return TupleIndex::Mask(mask.clone());
        }
    }
    if tuple.values.iter().any(|v| matches!(v, Value::DynamicNDArray(_))) {
        crate::helpers::composite::reject_runtime_shaped_operand();
    }
    TupleIndex::Axes(tuple.values.iter().cloned().map(SliceIndex::Single).collect())
}

/// Hand out a dynamic-bounds slice as a `DynamicNDArray`.
///
/// The slice is materialised into a buffer of the static envelope shape
/// whose content sits in the `[0, extent)` prefix of each axis, with the
/// runtime extents recorded in provenance. As a plain `StaticArray` that
/// buffer was consumed as if the envelope were the array: `a[0:r, :]`
/// fed `.mean()`, `.min()`, `len()`, element-wise ops and further indexing
/// with its zero padding as data. As a `DynamicNDArray` every consumer
/// sees the runtime shape. A slice without an extents record (a non-unit
/// step, or a complex dtype) has no runtime-shaped representation and is
/// rejected rather than compiled with its padding.
fn runtime_shaped_slice(b: &mut IRBuilder, padded: Value) -> Value {
    if let Value::StaticArray {
        segment_id,
        dtype,
        ..
    } = &padded
    {
        if *dtype != NumberType::Complex && b.provenance.runtime_extents_of(*segment_id).is_some() {
            return crate::helpers::promote::promote_static_to_dynamic_with_extents(b, &padded);
        }
    }
    panic!(
        "a slice with runtime bounds is supported only with step 1 on a real-valued array \
         (its result has a runtime shape)"
    )
}

#[allow(clippy::too_many_arguments)]
fn materialise_dynamic_axis_slice(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    axis: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let axis_len = shape[axis];
    let max_axis = axis_len;

    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }

    // Python's slice semantics (see `materialise_dynamic_1d_slice`).
    let (start_ir, stop_ir, step_ir, step_positive) = python_bounds(b, start, stop, step, axis_len);

    // Compile-time facts the defensive IR can elide:
    //   step_pos_known := step is provably > 0 (constant > 0 OR provable).
    //   start_nonneg   := start is provably >= 0 (a normalized start is in
    //                     [0, axis_len] for a positive step).
    // When both hold, src_axis_idx = start + slice_idx*step is provably >= 0
    // for slice_idx >= 0, so ge_zero / step_pos / gt_stop / is_neg / clamp_lo
    // are dead and we skip them. This is the overwhelmingly common
    // case (`a[i:j]` with `step is None`, `i, j` from min/max of indices).
    let step_pos_known: bool = match step_positive {
        Some(positive) => positive,
        None => provable_strictly_positive(b, &step_ir),
    };
    let start_nonneg: bool = match start_ir.int_val() {
        Some(s) => s >= 0,
        None => step_pos_known || provable_nonneg(b, &start_ir),
    };
    let step_is_one = step_ir.int_val() == Some(1);
    let fast = step_pos_known && start_nonneg;

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype);
    let zero = b.ir_constant_int(0);
    let axis_len_val = b.ir_constant_int(axis_len as i64);
    let axis_stride_val = b.ir_constant_int(strides[axis] as i64);

    let mut new_shape = shape.to_vec();
    new_shape[axis] = max_axis;
    let total: usize = new_shape.iter().product();
    let new_strides = row_major_strides(&new_shape);

    // Per-slice_idx computations (src_axis_idx, in_bounds, clamped) depend
    // only on the slice coordinate along the sliced axis, not on the other
    // axes' coordinates. Hoist them out of the per-cell loop so we pay the
    // mask + clamp cost `max_axis` times instead of `total` times. For 2D
    // bbox-style slices (e.g., A[ystart:ystop, xstart:xstop]) this is a
    // ~max_axis× savings on the slice mask phase (was the dominant cost on
    // ds1000_case0508 — ~400 comparisons → ~40).
    //
    // `in_bounds` is `None` when the slice index is provably in range at
    // build time (e.g., 0 <= start, stop = axis_len, both constants), so
    // the per-cell mask mux can be elided — Zinnia just reads through the
    // memory cell directly. This is what kills the 40K muxes on the patch-
    // reshape kernels 0388/0390 where the outer slice is structurally the
    // identity after `(N // p) * p == N`.
    let max_idx_val = b.ir_constant_int(axis_len as i64 - 1);
    // For slices whose stop is derived from a runtime witness (e.g.
    // `(N // patch_size) * patch_size`) the IR may carry tight bounds via
    // facts / range — try `ask_bounds` for an exact point before giving
    // up and emitting the runtime mask. Same for `start_ir`. This is what
    // closes the 0388/0390 patch-reshape: at trace time `patch_size = 10`
    // pins `stop` to 100 / 200 exactly.
    let stop_static = stop_ir.int_val().or_else(|| match b.ask_bounds(&stop_ir) {
        Some((lo, hi)) if lo == hi => Some(lo),
        _ => None,
    });
    let start_static = start_ir
        .int_val()
        .or_else(|| match b.ask_bounds(&start_ir) {
            Some((lo, hi)) if lo == hi => Some(lo),
            _ => None,
        });
    let mut per_idx: Vec<(Option<Value>, Value)> = Vec::with_capacity(max_axis);
    for slice_idx in 0..max_axis {
        let slice_idx_val = b.ir_constant_int(slice_idx as i64);
        let src_axis_idx = if step_is_one {
            b.ir_add_i(&start_ir, &slice_idx_val)
        } else {
            let off2 = b.ir_mul_i(&slice_idx_val, &step_ir);
            b.ir_add_i(&start_ir, &off2)
        };

        // Static "always in-bounds" elide: when both `src_axis_idx` and
        // `stop_ir` are comptime, decide the mask at build time and skip
        // emitting it for the in-bounds slot. Returns `None` to signal the
        // per-cell loop should bypass the mux entirely. Saves ~max_axis ×
        // (cells per axis) muxes when the slice covers the full source axis
        // (the 0388/0390 patch-reshape pattern after `(N//p)*p == N`).
        // Derive the comptime src axis position from `start_static +
        // slice_idx * step_static` when both are statically known. This
        // mirrors the runtime expression but reuses the resolved values.
        let step_static_for_idx = step_ir.int_val().unwrap_or(1);
        let static_src_axis_idx =
            start_static.map(|s| s + (slice_idx as i64) * step_static_for_idx);
        let static_in_bounds = match (static_src_axis_idx, stop_static) {
            (Some(si), Some(stop)) => {
                if si >= 0 && si < axis_len as i64 && si < stop {
                    Some(true)
                } else {
                    Some(false)
                }
            }
            _ => None,
        };
        let in_bounds: Option<Value> = if let Some(true) = static_in_bounds {
            None
        } else if let Some(false) = static_in_bounds {
            Some(b.ir_constant_bool(false))
        } else if fast {
            let lt_len = b.ir_less_than_i_bounded(&src_axis_idx, &axis_len_val);
            let lt_stop = b.ir_less_than_i_bounded(&src_axis_idx, &stop_ir);
            Some(b.ir_logical_and(&lt_len, &lt_stop))
        } else {
            let ge_zero = b.ir_greater_than_or_equal_i(&src_axis_idx, &zero);
            let lt_len = b.ir_less_than_i(&src_axis_idx, &axis_len_val);
            let in_range = b.ir_logical_and(&ge_zero, &lt_len);
            let lt_stop = b.ir_less_than_i(&src_axis_idx, &stop_ir);
            let stop_bool = if step_pos_known {
                lt_stop
            } else {
                let gt_stop = b.ir_greater_than_i(&src_axis_idx, &stop_ir);
                let step_pos = b.ir_greater_than_i(&step_ir, &zero);
                let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
                b.ir_bool_cast(&stop_ok)
            };
            Some(b.ir_logical_and(&in_range, &stop_bool))
        };

        let is_over = if start_nonneg && step_pos_known {
            b.ir_less_than_i_bounded(&max_idx_val, &src_axis_idx)
        } else {
            b.ir_greater_than_i(&src_axis_idx, &max_idx_val)
        };
        let clamped_hi = b.ir_select_i(&is_over, &max_idx_val, &src_axis_idx);
        let clamped = if start_nonneg && step_pos_known {
            clamped_hi
        } else {
            let is_neg = b.ir_less_than_i(&src_axis_idx, &zero);
            b.ir_select_i(&is_neg, &zero, &clamped_hi)
        };
        per_idx.push((in_bounds, clamped));
    }

    let mut leaves: Vec<Value> = Vec::with_capacity(total);
    for flat_out in 0..total {
        let coords = decode_coords(flat_out, &new_shape, &new_strides);
        let slice_idx = coords[axis];
        let (in_bounds, clamped) = per_idx[slice_idx].clone();

        let mut addr_static: i64 = offset as i64;
        for ax in 0..shape.len() {
            if ax == axis {
                continue;
            }
            addr_static += coords[ax] as i64 * strides[ax] as i64;
        }
        let other_offset = b.ir_constant_int(addr_static);
        let addr = if strides[axis] == 1 {
            b.ir_add_i(&other_offset, &clamped)
        } else {
            let axis_contrib = b.ir_mul_i(&clamped, &axis_stride_val);
            b.ir_add_i(&other_offset, &axis_contrib)
        };

        let elem = b.ir_read_memory(segment_id, &addr);
        let masked = match &in_bounds {
            None => elem,
            Some(cond) => {
                if dtype == NumberType::Float {
                    b.ir_select_f(cond, &elem, &default_val)
                } else {
                    b.ir_select_i(cond, &elem, &default_val)
                }
            }
        };
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&masked), dtype));
    }
    let out = super::base::build_static_array_from_flat(b, leaves, new_shape, dtype);
    // Runtime-extents provenance (padded layout: content sits in the
    // [0, extent) prefix of the sliced axis). step == 1 only — the bbox
    // family; other steps keep today's no-record behavior. Non-sliced
    // axes inherit the SOURCE's extents when recorded (slice-of-slice
    // composition), else their full static dims.
    if step_ir.int_val() == Some(1) {
        if let Value::StaticArray {
            segment_id: out_seg,
            ..
        } = &out
        {
            let diff = b.ir_sub_i(&stop_ir, &start_ir);
            let zero = b.ir_constant_int(0);
            let neg = b.ir_less_than_i(&diff, &zero);
            let extent = b.ir_select_i(&neg, &zero, &diff);
            if let Some(ext_vid) = extent.value_id() {
                let src_exts = b.provenance.runtime_extents_of(segment_id).cloned();
                let mut exts: Vec<crate::types::ValueId> = Vec::with_capacity(shape.len());
                let mut ok = true;
                for ax in 0..shape.len() {
                    if ax == axis {
                        exts.push(ext_vid);
                    } else if let Some(srcs) = src_exts.as_ref().and_then(|v| v.get(ax)) {
                        exts.push(*srcs);
                    } else {
                        let c = b.ir_constant_int(shape[ax] as i64);
                        match c.value_id() {
                            Some(vid) => exts.push(vid),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
                if ok {
                    b.provenance.set_runtime_extents(*out_seg, exts);
                }
            }
        }
    }
    out
}

/// Fuse the per-axis slice mask computation when 2+ axes have dynamic
/// bounds. Equivalent to calling `materialise_dynamic_axis_slice` once
/// per dynamic axis (recursing through `static_array_subscript`), but
/// avoids materialising the intermediate `max_axis × max_other` array
/// between axes — the AND of per-axis masks happens per output cell
/// against a single read of the original source.
///
/// `axis_specs` lists `(axis_idx, start, stop, step)` for each dynamic
/// Range axis (in increasing axis_idx order). Single-index and
/// static-Range axes have already been absorbed into `static_base` and
/// `out_shape` by the caller (mirroring `multidim_subscript_static_array`).
///
/// Called by `multidim_subscript_static_array` for the multi-axis dynamic
/// Range case (two-or-more dynamic Range axes with no remaining dynamic
/// single-index parts): it materialises all such axes in one pass instead
/// of the recursive per-axis path.
fn materialise_dynamic_multi_axis_slice(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    imag_seg: Option<u32>,
    strides: &[usize],
    static_base: i64,
    static_range_axes: &[(usize, usize, Vec<usize>)],
    axis_specs: &[(usize, usize, Option<Value>, Option<Value>, Option<Value>)],
    out_shape: &[usize],
) -> Value {
    if dtype != NumberType::Complex {
        super::base::materialize_static_array_segment(b, segment_id, dtype);
    }

    // Per dynamic axis: (src_axis_idx, out_axis_idx, start_ir, stop_ir,
    // step_ir, fast, step sign), the bounds by Python's slice semantics
    // (see `materialise_dynamic_1d_slice`).
    let mut axes: Vec<(usize, usize, Value, Value, Value, bool, Option<bool>)> =
        Vec::with_capacity(axis_specs.len());
    for (ax, out_ax, start, stop, step) in axis_specs {
        let (start_ir, stop_ir, step_ir, step_positive) =
            python_bounds(b, start.as_ref(), stop.as_ref(), step.as_ref(), shape[*ax]);
        let step_pos_known = match step_positive {
            Some(positive) => positive,
            None => provable_strictly_positive(b, &step_ir),
        };
        // A normalized start is in [0, axis_len] for a positive step.
        let start_nonneg = match start_ir.int_val() {
            Some(s) => s >= 0,
            None => step_pos_known || provable_nonneg(b, &start_ir),
        };
        let fast = step_pos_known && start_nonneg;
        let step_sign = if step_pos_known { Some(true) } else { step_positive };
        axes.push((*ax, *out_ax, start_ir, stop_ir, step_ir, fast, step_sign));
    }

    // Unit-step windows over every axis of a real array: a shift by a shared
    // one-hot per axis (`window_gather`) instead of a clamp, a box test and a
    // gather per output cell.
    if let Some(leaves) = window_gather(b, dtype, shape, segment_id, strides, static_base, static_range_axes, &axes, out_shape) {
        return finish_multi_axis_slice(b, leaves, out_shape, dtype, &axes);
    }

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype);
    let zero = b.ir_constant_int(0);

    // Per slice_idx along each axis: precompute (in_bounds_mask, clamped_idx).
    // Each axis's table has size = shape[axis]; shared across all output cells.
    let mut per_axis_table: Vec<Vec<(Option<Value>, Value)>> = Vec::with_capacity(axes.len());
    for (axis_idx, _, start_ir, stop_ir, step_ir, fast, step_sign) in &axes {
        let axis_len = shape[*axis_idx];
        let axis_len_val = b.ir_constant_int(axis_len as i64);
        let max_idx_val = b.ir_constant_int(axis_len as i64 - 1);
        let step_is_one = step_ir.int_val() == Some(1);
        let mut table: Vec<(Option<Value>, Value)> = Vec::with_capacity(axis_len);
        for slice_idx in 0..axis_len {
            let slice_idx_val = b.ir_constant_int(slice_idx as i64);
            let src_axis_idx = if step_is_one {
                b.ir_add_i(start_ir, &slice_idx_val)
            } else {
                let off2 = b.ir_mul_i(&slice_idx_val, step_ir);
                b.ir_add_i(start_ir, &off2)
            };
            let in_bounds: Option<Value> = if *fast {
                let lt_len = b.ir_less_than_i_bounded(&src_axis_idx, &axis_len_val);
                let lt_stop = b.ir_less_than_i_bounded(&src_axis_idx, stop_ir);
                Some(b.ir_logical_and(&lt_len, &lt_stop))
            } else {
                let ge_zero = b.ir_greater_than_or_equal_i(&src_axis_idx, &zero);
                let lt_len = b.ir_less_than_i(&src_axis_idx, &axis_len_val);
                let in_range = b.ir_logical_and(&ge_zero, &lt_len);
                // Before `stop` in the step's direction.
                let before_stop = match step_sign {
                    Some(true) => b.ir_less_than_i(&src_axis_idx, stop_ir),
                    Some(false) => b.ir_greater_than_i(&src_axis_idx, stop_ir),
                    None => {
                        let lt_stop = b.ir_less_than_i(&src_axis_idx, stop_ir);
                        let gt_stop = b.ir_greater_than_i(&src_axis_idx, stop_ir);
                        let step_pos = b.ir_greater_than_i(step_ir, &zero);
                        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
                        b.ir_bool_cast(&stop_ok)
                    }
                };
                Some(b.ir_logical_and(&in_range, &before_stop))
            };
            let is_over = if *fast {
                b.ir_less_than_i_bounded(&max_idx_val, &src_axis_idx)
            } else {
                b.ir_greater_than_i(&src_axis_idx, &max_idx_val)
            };
            let clamped_hi = b.ir_select_i(&is_over, &max_idx_val, &src_axis_idx);
            let clamped = if *fast {
                clamped_hi
            } else {
                let is_neg = b.ir_less_than_i(&src_axis_idx, &zero);
                b.ir_select_i(&is_neg, &zero, &clamped_hi)
            };
            table.push((in_bounds, clamped));
        }
        per_axis_table.push(table);
    }

    // Per-cell loop: AND the per-axis masks, compute address, read, mask.
    let total: usize = out_shape.iter().product();
    let new_strides = row_major_strides(out_shape);
    let mut leaves: Vec<Value> = Vec::with_capacity(total);
    for flat_out in 0..total {
        let out_coords = decode_coords(flat_out, out_shape, &new_strides);
        let mut combined_mask: Option<Value> = None;
        let mut addr = b.ir_constant_int(static_base);
        for (src_ax, out_ax, coords) in static_range_axes {
            let coord = coords[out_coords[*out_ax]];
            if coord != 0 {
                let contrib = b.ir_constant_int((coord * strides[*src_ax]) as i64);
                addr = b.ir_add_i(&addr, &contrib);
            }
        }
        for (table_idx, (axis_idx, out_ax, _, _, _, _, _)) in axes.iter().enumerate() {
            let (mask_k, clamped_k) = per_axis_table[table_idx][out_coords[*out_ax]].clone();
            if let Some(m) = mask_k {
                combined_mask = Some(match combined_mask {
                    None => m,
                    Some(prev) => b.ir_logical_and(&prev, &m),
                });
            }
            let scaled = if strides[*axis_idx] == 1 {
                clamped_k
            } else {
                let axis_stride_val = b.ir_constant_int(strides[*axis_idx] as i64);
                b.ir_mul_i(&clamped_k, &axis_stride_val)
            };
            addr = b.ir_add_i(&addr, &scaled);
        }
        if dtype == NumberType::Complex {
            let imag_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
            let elem = read_complex_leaf_dynamic(b, segment_id, imag_seg, &addr);
            let masked = match combined_mask {
                None => elem,
                Some(cond) => match elem {
                    Value::Complex { real, imag } => {
                        let (default_real, default_imag) = match default_val.clone() {
                            Value::Complex { real, imag } => (Value::Float(real), Value::Float(imag)),
                            _ => unreachable!(),
                        };
                        let real_v = b.ir_select_f(&cond, &Value::Float(real), &default_real);
                        let imag_v = b.ir_select_f(&cond, &Value::Float(imag), &default_imag);
                        match (real_v, imag_v) {
                            (Value::Float(real), Value::Float(imag)) => Value::Complex { real, imag },
                            _ => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                },
            };
            leaves.push(masked);
        } else {
            let elem = b.ir_read_memory(segment_id, &addr);
            let masked = match combined_mask {
                None => elem,
                Some(cond) => {
                    if dtype == NumberType::Float {
                        b.ir_select_f(&cond, &elem, &default_val)
                    } else {
                        b.ir_select_i(&cond, &elem, &default_val)
                    }
                }
            };
            leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&masked), dtype));
        }
    }
    if dtype == NumberType::Complex {
        let mut reals: Vec<Value> = Vec::with_capacity(leaves.len());
        let mut imags: Vec<Value> = Vec::with_capacity(leaves.len());
        for v in leaves {
            match v {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!(),
            }
        }
        return super::base::build_static_array_from_flat_complex(
            b,
            reals,
            imags,
            out_shape.to_vec(),
        );
    }
    finish_multi_axis_slice(b, leaves, out_shape, dtype, &axes)
}

/// The fused multi-axis slice's leaves as a static array, with the runtime
/// extents of its dynamic axes recorded (see `materialise_dynamic_multi_axis_slice`).
fn finish_multi_axis_slice(
    b: &mut IRBuilder,
    leaves: Vec<Value>,
    out_shape: &[usize],
    dtype: NumberType,
    axes: &[(usize, usize, Value, Value, Value, bool, Option<bool>)],
) -> Value {
    let out = super::base::build_static_array_from_flat(b, leaves, out_shape.to_vec(), dtype);
    // Runtime-extents provenance for the fused multi-axis slice: each
    // dynamic axis (step 1) gets extent = max(0, stop - start); static
    // Range axes occupy their full output dims. Bail (no record) when
    // any dynamic axis has a non-unit step.
    if let Value::StaticArray {
        segment_id: out_seg,
        ..
    } = &out
    {
        let mut by_out_axis: Vec<Option<crate::types::ValueId>> = vec![None; out_shape.len()];
        let mut ok = true;
        for (_, out_axis, s_ir, e_ir, step_ir, _, _) in axes {
            if step_ir.int_val() != Some(1) {
                ok = false;
                break;
            }
            let diff = b.ir_sub_i(e_ir, s_ir);
            let zero = b.ir_constant_int(0);
            let neg = b.ir_less_than_i(&diff, &zero);
            let extent = b.ir_select_i(&neg, &zero, &diff);
            match extent.value_id() {
                Some(vid) if *out_axis < by_out_axis.len() => by_out_axis[*out_axis] = Some(vid),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let mut exts: Vec<crate::types::ValueId> = Vec::with_capacity(out_shape.len());
            for (oa, slot) in by_out_axis.into_iter().enumerate() {
                match slot {
                    Some(vid) => exts.push(vid),
                    None => {
                        let c = b.ir_constant_int(out_shape[oa] as i64);
                        match c.value_id() {
                            Some(vid) => exts.push(vid),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                }
            }
            if ok {
                b.provenance.set_runtime_extents(*out_seg, exts);
            }
        }
    }
    out
}

/// Prototype (costgap/index.md, section 4.1): `x[s0:e0, s1:e1, ...]` with unit
/// steps and runtime bounds on every axis of a real array, as a shift of the
/// whole array by a one-hot of each start, one axis at a time:
/// `y[.., u, ..] = sum_d sel[u][d] * x[.., u + d, ..]` with
/// `sel[u][d] = [start == d] and [start + u < stop]`. Terms with
/// `u + d >= n` are dropped, so positions past the array end read 0, and the
/// stop mask zeroes positions past the window, which is the default 0 the
/// per-cell lowering writes there. Every factor is computed (no hint): the
/// equality bits have at most one 1 (`start` is in `[0, n]` on this path),
/// so each sum is `x[.., start + u, ..]` or 0. `None` when the slice is not of
/// that form.
#[allow(clippy::too_many_arguments)]
fn window_gather(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    static_base: i64,
    static_range_axes: &[(usize, usize, Vec<usize>)],
    axes: &[(usize, usize, Value, Value, Value, bool, Option<bool>)],
    out_shape: &[usize],
) -> Option<Vec<Value>> {
    let all_axes_windowed = dtype != NumberType::Complex
        && static_range_axes.is_empty()
        && axes.len() == shape.len()
        && out_shape == shape
        && axes.iter().enumerate().all(|(k, (ax, out_ax, _, _, step, fast, _))| {
            *ax == k && *out_ax == k && *fast && step.int_val() == Some(1)
        });
    if !all_axes_windowed || static_base < 0 {
        return None;
    }
    // cells[flat] over the current (partially shifted) array, row-major in `shape`.
    let row_strides = row_major_strides(shape);
    let total: usize = shape.iter().product();
    let mut cells: Vec<Value> = (0..total)
        .map(|flat| {
            let coords = decode_coords(flat, shape, &row_strides);
            let addr: usize = coords.iter().zip(strides).map(|(c, s)| c * s).sum();
            read_leaf_at_flat(b, segment_id, static_base as usize, addr, dtype)
        })
        .collect();
    for (ax, _, start, stop, _, _, _) in axes {
        let n = shape[*ax];
        let hits: Vec<Value> = (0..n)
            .map(|d| {
                let d_v = b.ir_constant_int(d as i64);
                b.ir_equal_i(start, &d_v)
            })
            .collect();
        let sel: Vec<Vec<Value>> = (0..n)
            .map(|u| {
                let u_v = b.ir_constant_int(u as i64);
                let at = b.ir_add_i(start, &u_v);
                let inside = b.ir_less_than_i_bounded(&at, stop);
                hits[..n - u].iter().map(|h| b.ir_logical_and(h, &inside)).collect()
            })
            .collect();
        let mut next = Vec::with_capacity(total);
        for flat in 0..total {
            let coords = decode_coords(flat, shape, &row_strides);
            let u = coords[*ax];
            let mut acc: Option<Value> = None;
            for (d, s) in sel[u].iter().enumerate() {
                let src = flat + d * row_strides[*ax];
                let product = b.create_ir(&crate::ir_defs::IR::MulI, &[s.clone(), cells[src].clone()]);
                let term = match (product, dtype) {
                    (Value::Integer(sv), NumberType::Float) => Value::Float(crate::types::ScalarValue {
                        static_val: None,
                        stmt_id: sv.stmt_id,
                        value_id: sv.value_id,
                    }),
                    (product, _) => product,
                };
                acc = Some(match acc {
                    None => term,
                    Some(prev) if dtype == NumberType::Float => b.ir_add_f(&prev, &term),
                    Some(prev) => b.ir_add_i(&prev, &term),
                });
            }
            next.push(acc.expect("u < n, so at least one term"));
        }
        cells = next;
    }
    Some(cells.into_iter().map(|v| scalar_i64_to_value(&value_to_scalar_i64(&v), dtype)).collect())
}

fn provable_nonneg(b: &IRBuilder, v: &Value) -> bool {
    let Some(vid) = v.value_id() else {
        return false;
    };
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    matches!(b.prove(&term), ProveOutcome::Proved)
}

fn provable_strictly_positive(b: &IRBuilder, v: &Value) -> bool {
    let Some(vid) = v.value_id() else {
        return false;
    };
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    let term = ContractTerm::Cmp {
        op: CmpOp::Gt,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    matches!(b.prove(&term), ProveOutcome::Proved)
}

/// Materialise a single dynamic-axis-0 row into a fresh segment. For Complex
/// dtype, two parallel reads/writes per element across the dual segments.
/// `idx` must already be wrapped and bounds-checked (`checked_axis_index`).
fn materialise_axis0_dynamic_row(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    imag_seg: Option<u32>,
    strides: &[usize],
    offset: usize,
    idx: &Value,
) -> Value {
    let row_stride = strides[0];
    let row_shape = shape[1..].to_vec();
    let row_total: usize = row_shape.iter().product();
    let stride_val = b.ir_constant_int(row_stride as i64);
    let offset_val = b.ir_constant_int(offset as i64);
    let base = b.ir_mul_i(idx, &stride_val);
    let base = b.ir_add_i(&offset_val, &base);

    if dtype == NumberType::Complex {
        let imag_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
        super::base::materialize_static_array_segment(b, segment_id, NumberType::Float);
        super::base::materialize_static_array_segment(b, imag_seg, NumberType::Float);
        let mut real_leaves: Vec<Value> = Vec::with_capacity(row_total);
        let mut imag_leaves: Vec<Value> = Vec::with_capacity(row_total);
        for j in 0..row_total {
            let off2 = b.ir_constant_int(j as i64);
            let addr = b.ir_add_i(&base, &off2);
            let r = b.ir_read_memory(segment_id, &addr);
            let im = b.ir_read_memory(imag_seg, &addr);
            real_leaves.push(scalar_i64_to_value(
                &value_to_scalar_i64(&r),
                NumberType::Float,
            ));
            imag_leaves.push(scalar_i64_to_value(
                &value_to_scalar_i64(&im),
                NumberType::Float,
            ));
        }
        return super::base::build_static_array_from_flat_complex(
            b,
            real_leaves,
            imag_leaves,
            row_shape,
        );
    }

    super::base::materialize_static_array_segment(b, segment_id, dtype);
    let mut leaves: Vec<Value> = Vec::with_capacity(row_total);
    for j in 0..row_total {
        let off2 = b.ir_constant_int(j as i64);
        let addr = b.ir_add_i(&base, &off2);
        let elem = b.ir_read_memory(segment_id, &addr);
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&elem), dtype));
    }
    super::base::build_static_array_from_flat(b, leaves, row_shape, dtype)
}

/// Multi-axis subscript with mixed Single + Range. This is the general case;
/// covers `arr[i, j]`, `arr[:, j]`, `arr[a:b, c]`, etc.
fn multidim_subscript_static_array(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
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
        _ => unreachable!(),
    };

    let rank = shape.len();
    if indices.len() > rank {
        // Trailing indices can only be NewAxis (handled at top-level) — at
        // this depth, treat the rest as scalar-trailing, which means too
        // many indices.
        panic!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            rank,
            indices.len()
        );
    }

    // If every position is Single and resolved (covered earlier), so reach
    // here for at least one Range. Compute a per-axis "selection" — either
    // Single (consumes the axis) or Range coords.
    let mut out_shape: Vec<usize> = Vec::new();
    let mut axis_kind: Vec<AxisSel> = Vec::with_capacity(rank);
    let mut out_axis_of_src: Vec<Option<usize>> = vec![None; rank];

    for ax in 0..rank {
        if ax >= indices.len() {
            // Trailing axes default to full range.
            out_shape.push(shape[ax]);
            out_axis_of_src[ax] = Some(out_shape.len() - 1);
            axis_kind.push(AxisSel::FullRange);
            continue;
        }
        match &indices[ax] {
            SliceIndex::Single(v) => {
                axis_kind.push(AxisSel::Single(v.clone()));
            }
            SliceIndex::Range(s, e, st) => {
                let len_i = shape[ax] as i64;
                let is_absent =
                    |v: &Option<Value>| -> bool { matches!(v.as_ref(), Some(Value::None) | None) };
                let int_of = |v: &Option<Value>| -> Option<i64> {
                    match v.as_ref() {
                        Some(Value::None) | None => None,
                        Some(val) => val.int_val(),
                    }
                };
                let st_absent = is_absent(st);
                let s_absent = is_absent(s);
                let e_absent = is_absent(e);
                let st_int = int_of(st);
                let s_int = int_of(s);
                let e_int = int_of(e);
                let st_ok = st_absent || st_int.is_some();
                let s_ok = s_absent || s_int.is_some();
                let e_ok = e_absent || e_int.is_some();
                if st_ok && s_ok && e_ok {
                    let stst = st_int.unwrap_or(1);
                    assert!(stst != 0, "slice step cannot be zero");
                    // Pythonic defaults depend on step sign.
                    let ss = if s_absent {
                        if stst > 0 {
                            0
                        } else {
                            len_i - 1
                        }
                    } else {
                        let raw = s_int.unwrap();
                        if raw < 0 {
                            (len_i + raw).max(0)
                        } else {
                            raw.min(len_i)
                        }
                    };
                    let ee = if e_absent {
                        if stst > 0 {
                            len_i
                        } else {
                            -1
                        }
                    } else {
                        let raw = e_int.unwrap();
                        if raw < 0 {
                            (len_i + raw).max(0)
                        } else {
                            raw.min(len_i)
                        }
                    };
                    let mut coords: Vec<usize> = Vec::new();
                    if stst > 0 {
                        let mut i = ss;
                        while i < ee {
                            coords.push(i as usize);
                            i += stst;
                        }
                    } else {
                        let mut i = ss;
                        while i > ee {
                            coords.push(i as usize);
                            i += stst;
                        }
                    }
                    out_shape.push(coords.len());
                    out_axis_of_src[ax] = Some(out_shape.len() - 1);
                    axis_kind.push(AxisSel::Coords(coords));
                } else {
                    out_shape.push(shape[ax]);
                    out_axis_of_src[ax] = Some(out_shape.len() - 1);
                    axis_kind.push(AxisSel::DynamicRange(
                        s.clone(),
                        e.clone(),
                        st.clone(),
                    ));
                }
            }
            _ => unreachable!("Ellipsis/NewAxis already handled at top-level"),
        }
    }

    // Compute base from Single static contributions; collect dynamic Single
    // contributions; collect axis-coord arrays for Range axes.
    let mut static_base: i64 = offset as i64;
    let mut dynamic_parts: Vec<Value> = Vec::new();
    let mut dynamic_single_axes: Vec<(usize, Value)> = Vec::new();
    let mut range_axes: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut dynamic_range_axes: Vec<(usize, usize, Option<Value>, Option<Value>, Option<Value>)> =
        Vec::new();

    for ax in 0..rank {
        match &axis_kind[ax] {
            AxisSel::Single(v) => {
                // Wrap and bounds-check the index on its own axis, once,
                // so the flat address (`dynamic_parts`, consumed by the
                // memory-read fallback and Complex path) and the
                // coordinate tests of the payload gather agree for a
                // negative `i` (C12c), and every leaf address below lies
                // inside the array without a per-leaf check.
                let vn = checked_axis_index(b, v, ax, shape[ax], "static_array_read");
                if let Some(i) = vn.int_val() {
                    static_base += i * strides[ax] as i64;
                } else {
                    let stride_val = b.ir_constant_int(strides[ax] as i64);
                    dynamic_parts.push(b.ir_mul_i(&vn, &stride_val));
                    dynamic_single_axes.push((ax, vn));
                }
            }
            AxisSel::Coords(coords) => {
                range_axes.push((ax, coords.clone()));
            }
            AxisSel::FullRange => {
                let coords: Vec<usize> = (0..shape[ax]).collect();
                range_axes.push((ax, coords));
            }
            AxisSel::DynamicRange(start, stop, step) => {
                dynamic_range_axes.push((
                    ax,
                    out_axis_of_src[ax].expect("dynamic range axis must map to an output axis"),
                    start.clone(),
                    stop.clone(),
                    step.clone(),
                ));
            }
        }
    }

    if !dynamic_range_axes.is_empty() {
        if dynamic_range_axes.len() >= 2 && dynamic_parts.is_empty() {
            let mut static_range_axes: Vec<(usize, usize, Vec<usize>)> =
                Vec::with_capacity(range_axes.len());
            for (src_ax, coords) in &range_axes {
                static_range_axes.push((
                    *src_ax,
                    out_axis_of_src[*src_ax].expect("range axis must map to an output axis"),
                    coords.clone(),
                ));
            }
            let padded = materialise_dynamic_multi_axis_slice(
                b,
                dtype,
                &shape,
                segment_id,
                imag_seg,
                &strides,
                static_base,
                &static_range_axes,
                &dynamic_range_axes,
                &out_shape,
            );
            return runtime_shaped_slice(b, padded);
        }

        // Preserve the existing recursive single-axis materialisation path
        // when there is only one dynamic range axis or when a dynamic Single
        // index contributes to the address; the recursive subscript below
        // checks the remaining Single indices on their own axes.
        let (ax, _, start, stop, step) = dynamic_range_axes[0].clone();
        let sliced = slice_axis(
            b,
            val,
            dtype,
            &shape,
            segment_id,
            &strides,
            offset,
            ax,
            start.as_ref(),
            stop.as_ref(),
            step.as_ref(),
        );
        let remaining: Vec<SliceIndex> = indices
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                if i == ax {
                    SliceIndex::Range(None, None, None)
                } else {
                    idx.clone()
                }
            })
            .collect();
        // Identity subscript: every remaining axis is a bare full range
        // (the `data[:, :k]` pattern). Re-subscripting would copy the
        // materialised slice into a fresh segment — wasted memory ops,
        // and the copy would shed the slice's runtime-extents record.
        // Bounds may be parsed as concrete 0..len step 1 rather than
        // bare `None`s, so test full-ness against the axis length.
        let full_range = |idx: &SliceIndex, len: usize| match idx {
            SliceIndex::Range(s, e, st) => {
                let unset = |v: &Value| matches!(v, Value::None);
                s.as_ref().map_or(true, |v| unset(v) || v.int_val() == Some(0))
                    && e.as_ref()
                        .map_or(true, |v| unset(v) || v.int_val() == Some(len as i64))
                    && st.as_ref().map_or(true, |v| unset(v) || v.int_val() == Some(1))
            }
            _ => false,
        };
        if remaining.len() == shape.len()
            && remaining
                .iter()
                .zip(shape.iter())
                .all(|(idx, &len)| full_range(idx, len))
        {
            return sliced;
        }
        return match &sliced {
            Value::DynamicNDArray(d) => crate::helpers::array_ops::dyn_subscript(b, d, &remaining),
            _ => static_array_subscript(b, &sliced, &remaining),
        };
    }

    let total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    // Special case: out_shape empty (all axes were Single) — that's the
    // all-single fast path covered earlier; should not reach here. Guard.
    if total == 0 || range_axes.is_empty() {
        // All Single but indices.len() < rank → out is a view into the
        // remaining axes. Build a view StaticArray.
        if range_axes.is_empty() && indices.len() < rank {
            // Shouldn't happen — we filled trailing axes as FullRange.
            unreachable!();
        }
        // Empty dimension somewhere → empty array.
        let empty: Vec<Value> = Vec::new();
        return super::base::build_static_array_from_flat(b, empty, out_shape, dtype);
    }

    let gather_axes: Vec<GatherAxis> = dynamic_single_axes
        .iter()
        .map(|(ax, index)| GatherAxis { index: index.clone(), stride: strides[*ax], len: shape[*ax] })
        .collect();
    let mut gather = if PayloadGather::affordable(&gather_axes) {
        PayloadGather::of_segment(b, segment_id, dtype)
    } else {
        None
    };
    let mut leaves: Vec<Value> = Vec::with_capacity(total);
    for flat_out in 0..total {
        let out_coords = decode_coords(flat_out, &out_shape, &out_strides);
        let mut src_flat = static_base;
        for (out_ax, (src_ax, coords)) in range_axes.iter().enumerate() {
            src_flat += coords[out_coords[out_ax]] as i64 * strides[*src_ax] as i64;
        }
        if dynamic_parts.is_empty() {
            // Fully static address (after relative-to-segment normalisation).
            // src_flat is already absolute (offset baked into static_base).
            let flat_idx = src_flat as usize;
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf(b, segment_id, im, flat_idx));
            } else {
                // read_leaf_at_flat takes a base+rel; pass abs as 0+abs.
                leaves.push(read_leaf_at_flat(b, segment_id, 0, flat_idx, dtype));
            }
        } else if let Some(v) = gather.as_mut().and_then(|g| g.read(b, src_flat, &gather_axes)) {
            leaves.push(v);
        } else {
            let static_part = b.ir_constant_int(src_flat);
            let mut acc = static_part;
            for p in &dynamic_parts {
                acc = b.ir_add_i(&acc, p);
            }
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf_dynamic(b, segment_id, im, &acc));
            } else {
                leaves.push(read_leaf_at_checked_address(b, segment_id, &acc, dtype));
            }
        }
    }
    if dtype == NumberType::Complex {
        let mut reals: Vec<Value> = Vec::with_capacity(leaves.len());
        let mut imags: Vec<Value> = Vec::with_capacity(leaves.len());
        for v in leaves {
            match v {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!(),
            }
        }
        return super::base::build_static_array_from_flat_complex(b, reals, imags, out_shape);
    }
    super::base::build_static_array_from_flat(b, leaves, out_shape, dtype)
}

enum AxisSel {
    Single(Value),
    Coords(Vec<usize>),
    FullRange,
    DynamicRange(Option<Value>, Option<Value>, Option<Value>),
}

// ────────────────────────────────────────────────────────────────────────
// Iteration helpers
// ────────────────────────────────────────────────────────────────────────

/// Build the per-iteration value for `for x in arr` over a `Value::StaticArray`.
/// For a 1-D array this is a leaf (Integer / Float / Boolean) read at index
/// `i`. For a ≥2-D array this is a (D-1)-rank `Value::StaticArray` view
/// sharing the same segment.
pub fn iter_element(b: &mut IRBuilder, arr: &Value, i: usize) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match arr {
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
        _ => panic!("iter_element: expected StaticArray"),
    };
    if shape.len() == 1 {
        // For Complex 1-D, return Value::Complex {real, imag}.
        if dtype == NumberType::Complex {
            return read_complex_leaf(
                b,
                segment_id,
                imag_seg.expect("Complex array missing imag_segment_id"),
                offset + i * strides[0],
            );
        }
        return read_leaf_at_flat(b, segment_id, offset, i * strides[0], dtype);
    }
    // ≥2-D → view sharing segment(s).
    let new_shape = shape[1..].to_vec();
    let new_strides = strides[1..].to_vec();
    let new_offset = offset + i * strides[0];
    Value::StaticArray {
        dtype,
        shape: new_shape,
        segment_id,
        strides: new_strides,
        offset: new_offset,
        imag_segment_id: imag_seg,
        value_id: ValueId::next(),
    }
}

/// Read a single Complex element from a dual-segment Complex StaticArray at
/// the given absolute flat offset. Prefers cached Value::Complex wires.
pub fn read_complex_leaf(
    b: &mut IRBuilder,
    real_seg: u32,
    imag_seg: u32,
    abs_offset: usize,
) -> Value {
    if let Some(cached) = b.static_array_payload.get(&real_seg) {
        if abs_offset < cached.len() {
            // Cache (rep A) holds Value::Complex directly.
            return cached[abs_offset].clone();
        }
    }
    let addr = b.ir_constant_int(abs_offset as i64);
    super::base::materialize_static_array_segment(b, real_seg, NumberType::Float);
    super::base::materialize_static_array_segment(b, imag_seg, NumberType::Float);
    let real_raw = b.ir_read_memory(real_seg, &addr);
    let imag_raw = b.ir_read_memory(imag_seg, &addr);
    let real_sv = match scalar_i64_to_value(&value_to_scalar_i64(&real_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let imag_sv = match scalar_i64_to_value(&value_to_scalar_i64(&imag_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    Value::Complex {
        real: real_sv,
        imag: imag_sv,
    }
}

/// Read a Complex element at a runtime address (sum of `offset + sum_k(idx*stride)`).
pub fn read_complex_leaf_dynamic(
    b: &mut IRBuilder,
    real_seg: u32,
    imag_seg: u32,
    addr: &Value,
) -> Value {
    super::base::materialize_static_array_segment(b, real_seg, NumberType::Float);
    super::base::materialize_static_array_segment(b, imag_seg, NumberType::Float);
    let real_raw = b.ir_read_memory(real_seg, addr);
    let imag_raw = b.ir_read_memory(imag_seg, addr);
    let real_sv = match scalar_i64_to_value(&value_to_scalar_i64(&real_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let imag_sv = match scalar_i64_to_value(&value_to_scalar_i64(&imag_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    Value::Complex {
        real: real_sv,
        imag: imag_sv,
    }
}
