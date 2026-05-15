//! Native shape-manipulation paths for `Value::StaticArray`.
//!
//! P4c of `compiler.epic-segment-native-static-arrays`: this module replaces
//! the boundary shim that used to convert a `StaticArray` into a nested
//! `Value::List` before dispatching to the legacy machinery in
//! `helpers::array_ops::shape`, `helpers::ndarray::ndarray_transpose`, and
//! `ops::static_ndarray_ops::{ndarray_reshape, ndarray_moveaxis,
//! np_concatenate, np_stack, np_hstack, np_vstack, np_column_stack,
//! np_squeeze, np_expand_dims}`.
//!
//! View vs. materialise policy (per op):
//!
//! - **`reshape` to a fully-static target shape**: pure metadata. Same
//!   `segment_id`, new `shape`, new contiguous row-major `strides`, `offset`
//!   preserved. Cache entry reused.
//! - **`transpose` / `.T` / `moveaxis`**: materialise. Mirrors
//!   `dyn_ndarray::reshape::dyn_transpose`'s rationale — a stride-only view
//!   would complicate downstream non-contiguous-aware ops, so we always
//!   write a fresh contiguous segment.
//! - **`concatenate` / `stack` / `hstack` / `vstack` / `column_stack`**:
//!   allocate a fresh segment of the combined size, copy each input cell
//!   in. Output is a fresh `Value::StaticArray` with a freshly-cached
//!   payload.
//! - **`expand_dims` / `squeeze`**: pure metadata (insert / remove length-1
//!   dims). Same segment + cache.
//! - **`flatten` / `ravel`**: when the source is contiguous (`offset==0`,
//!   row-major strides matching `shape`), pure metadata — flat 1-D view.
//!   When the source is a non-contiguous view (rare for StaticArray after
//!   P2's "materialise on non-contiguous slice" policy), materialise into a
//!   fresh segment.
//!
//! Cache awareness: every op reads via the `payload_cells` helper from the
//! P4a module, which uses `IRBuilder::static_array_payload` when present and
//! falls back to `ir_read_memory` if the cache was invalidated by a P3
//! dynamic write. New outputs go through `build_static_array_from_flat`,
//! which populates the cache with the per-cell wires that carry through
//! constant folding.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value, ValueId};

use super::super::shape_arith::{decode_coords, row_major_strides};
use super::base::build_static_array_from_flat;
use super::elementwise::payload_cells;

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn dtype_of(val: &Value) -> NumberType {
    match val {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => panic!("dtype_of: expected StaticArray"),
    }
}

fn shape_of(val: &Value) -> Vec<usize> {
    match val {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => panic!("shape_of: expected StaticArray"),
    }
}

/// Is this StaticArray a contiguous row-major view of its segment? If so we
/// can do reshape / flatten as pure metadata; otherwise we must materialise.
fn is_contiguous(val: &Value) -> bool {
    match val {
        Value::StaticArray { shape, strides, offset, .. } => {
            *offset == 0 && *strides == row_major_strides(shape)
        }
        _ => false,
    }
}

/// Materialise a (possibly non-contiguous) StaticArray view into a fresh
/// contiguous payload (a `Vec<Value>` flat in row-major order over the
/// view's `shape`). Reuses cached wires when the source has no offset and
/// the strides match contiguous; otherwise walks via `strides`/`offset` and
/// rebuilds. The returned vector has length `prod(shape)` and is suitable
/// for handing to `build_static_array_from_flat`.
fn materialise_to_flat(b: &mut IRBuilder, val: &Value) -> Vec<Value> {
    // payload_cells handles both contiguous and (less common) non-contiguous
    // cases — but it walks over `offset..offset+total` linearly, which is
    // only correct for contiguous views. Compute strided cell reads by hand
    // when the view is non-contiguous.
    let (shape, strides, offset, _segment_id, dtype) = match val {
        Value::StaticArray { shape, strides, offset, segment_id, dtype, .. } => {
            (shape.clone(), strides.clone(), *offset, *segment_id, *dtype)
        }
        _ => panic!("materialise_to_flat: expected StaticArray"),
    };
    let _ = dtype;
    if is_contiguous(val) || offset == 0 && strides == row_major_strides(&shape) {
        return payload_cells(b, val);
    }
    // Non-contiguous view: walk shape in row-major order, using strides to
    // recover the absolute payload index.
    let total: usize = shape.iter().product();
    let logical_strides = row_major_strides(&shape);
    let mut out: Vec<Value> = Vec::with_capacity(total);
    for flat in 0..total {
        let coords = decode_coords(flat, &shape, &logical_strides);
        let mut abs: usize = offset;
        for (ax, &c) in coords.iter().enumerate() {
            abs += c * strides[ax];
        }
        // Reuse payload_cells via a synthetic single-cell view: but the
        // simplest is to read the cached value at `abs` directly.
        let segment_id = match val {
            Value::StaticArray { segment_id, .. } => *segment_id,
            _ => unreachable!(),
        };
        let cell = if let Some(cached) = b.static_array_payload.get(&segment_id) {
            if abs < cached.len() {
                cached[abs].clone()
            } else {
                let addr = b.ir_constant_int(abs as i64);
                let raw = b.ir_read_memory(segment_id, &addr);
                let dtype = dtype_of(val);
                crate::ops::dyn_ndarray::scalar_i64_to_value(
                    &crate::ops::dyn_ndarray::value_to_scalar_i64(&raw),
                    dtype,
                )
            }
        } else {
            let addr = b.ir_constant_int(abs as i64);
            let raw = b.ir_read_memory(segment_id, &addr);
            let dtype = dtype_of(val);
            crate::ops::dyn_ndarray::scalar_i64_to_value(
                &crate::ops::dyn_ndarray::value_to_scalar_i64(&raw),
                dtype,
            )
        };
        out.push(cell);
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
// Reshape
// ────────────────────────────────────────────────────────────────────────

/// Parse a reshape target shape from `args`. `args` is either a single
/// tuple/list of ints, a single int, or multiple int positional args.
/// Supports `-1` inference for one dim. Returns `None` if any element is
/// not statically known.
fn parse_reshape_target(args: &[Value], total: usize) -> Option<Vec<usize>> {
    // Collect raw i64 values (preserving sign). One -1 is the inference
    // sentinel.
    let raw: Vec<i64> = if args.len() == 1 {
        match &args[0] {
            Value::Tuple(d) | Value::List(d) => {
                let mut out = Vec::with_capacity(d.values.len());
                for v in &d.values {
                    out.push(v.int_val()?);
                }
                out
            }
            Value::Integer(_) => vec![args[0].int_val()?],
            _ => return None,
        }
    } else {
        let mut out = Vec::with_capacity(args.len());
        for v in args {
            out.push(v.int_val()?);
        }
        out
    };

    let neg_count = raw.iter().filter(|&&n| n == -1).count();
    if neg_count > 1 {
        panic!("reshape: can only specify one unknown dimension");
    }
    if neg_count == 1 {
        let known_product: i64 = raw.iter().filter(|&&n| n != -1).product();
        if known_product <= 0 {
            return None;
        }
        if total as i64 % known_product != 0 {
            panic!(
                "reshape: cannot reshape array of size {} into shape with -1 inference",
                total
            );
        }
        let inferred = total as i64 / known_product;
        Some(
            raw.iter()
                .map(|&n| if n == -1 { inferred as usize } else { n as usize })
                .collect(),
        )
    } else {
        Some(raw.iter().map(|&n| n as usize).collect())
    }
}

/// Try to apply `reshape` natively. Returns `Some(view-or-copy)` for the
/// migrated all-static-target case; `None` otherwise (caller should fall
/// back to the legacy path).
pub fn try_apply_reshape(
    b: &mut IRBuilder,
    val: &Value,
    args: &[Value],
) -> Option<Value> {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => return None,
    };
    let total: usize = shape.iter().product();
    let target = parse_reshape_target(args, total)?;
    let target_total: usize = target.iter().product();
    if target_total != total {
        panic!(
            "reshape: cannot reshape array of size {} into shape {:?}",
            total, target
        );
    }
    // Pure metadata for contiguous views. For non-contiguous ones (rare),
    // materialise into a fresh segment first.
    let out = if is_contiguous(&Value::StaticArray {
        dtype,
        shape: shape.clone(),
        segment_id,
        strides: strides.clone(),
        offset,
        imag_segment_id: imag_seg,
        value_id: ValueId::next(),
    }) {
        let new_strides = row_major_strides(&target);
        Value::StaticArray {
            dtype,
            shape: target,
            segment_id,
            strides: new_strides,
            offset,
            imag_segment_id: imag_seg,
            value_id: ValueId::next(),
        }
    } else if dtype == NumberType::Complex {
        // Non-contiguous view: materialise into a fresh segment.
        // Component-wise materialise. payload_cells/materialise_to_flat
        // returns Value::Complex cells; split, then rebuild.
        let cells = materialise_to_flat(b, val);
        let mut reals = Vec::with_capacity(cells.len());
        let mut imags = Vec::with_capacity(cells.len());
        for c in cells {
            match c {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!("Complex StaticArray cell expected"),
            }
        }
        crate::helpers::static_array::base::build_static_array_from_flat_complex(b, reals, imags, target)
    } else {
        let flat = materialise_to_flat(b, val);
        build_static_array_from_flat(b, flat, target, dtype)
    };
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    Some(out)
}

// ────────────────────────────────────────────────────────────────────────
// Transpose / moveaxis
// ────────────────────────────────────────────────────────────────────────

/// Resolve the transpose permutation from `args`. `args` is either:
///   - empty / `Value::None` → reverse axes (`[ndim-1, ..., 0]`)
///   - a single tuple/list of ints
///   - multiple int positional args
fn parse_transpose_perm(args: &[Value], ndim: usize) -> Option<Vec<usize>> {
    let raw: Vec<i64> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
        return Some((0..ndim).rev().collect());
    } else if let Some(Value::Tuple(d)) | Some(Value::List(d)) = args.first() {
        let mut out = Vec::with_capacity(d.values.len());
        for v in &d.values {
            out.push(v.int_val()?);
        }
        out
    } else {
        let mut out = Vec::with_capacity(args.len());
        for v in args {
            out.push(v.int_val()?);
        }
        out
    };
    if raw.len() != ndim {
        panic!(
            "transpose: axes length {} does not match array rank {}",
            raw.len(),
            ndim
        );
    }
    let mut perm: Vec<usize> = Vec::with_capacity(ndim);
    for a in raw {
        let resolved = if a < 0 { ndim as i64 + a } else { a };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!(
                "transpose: axis {} is out of bounds for array of rank {}",
                a, ndim
            );
        }
        perm.push(resolved as usize);
    }
    let mut seen = vec![false; ndim];
    for &p in &perm {
        if seen[p] {
            panic!("transpose: axes must be a permutation of 0..{}", ndim);
        }
        seen[p] = true;
    }
    Some(perm)
}

/// Compute a transposed StaticArray by materialising into a fresh segment.
/// Mirrors `dyn_transpose`'s policy.
fn transpose_materialise(b: &mut IRBuilder, val: &Value, perm: &[usize]) -> Value {
    let (dtype, shape, _segment_id, strides, offset) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, .. } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset)
        }
        _ => unreachable!(),
    };
    let ndim = shape.len();
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_total: usize = new_shape.iter().product();
    let new_strides_for_decode = row_major_strides(&new_shape);

    // Read source via the cache or per-cell. Use payload_cells when
    // contiguous (it grabs the cached window); otherwise fall back to
    // strided reads via materialise_to_flat (which is contiguous-flat).
    // For transpose we want addressing by source coords with source strides,
    // so we read raw cells in flat-source order and index by the source's
    // logical_strides over `shape`. When the view is non-contiguous, we
    // first build a contiguous flat copy.
    let src_flat: Vec<Value> = if offset == 0 && strides == row_major_strides(&shape) {
        payload_cells(b, val)
    } else {
        materialise_to_flat(b, val)
    };
    let src_strides = row_major_strides(&shape);

    let mut out: Vec<Value> = Vec::with_capacity(new_total);
    for flat_out in 0..new_total {
        let out_coords = decode_coords(flat_out, &new_shape, &new_strides_for_decode);
        // out_coords[i] is along axis perm[i] of the source.
        let mut src_lin: usize = 0;
        for i in 0..ndim {
            src_lin += out_coords[i] * src_strides[perm[i]];
        }
        out.push(src_flat[src_lin].clone());
    }
    if dtype == NumberType::Complex {
        let mut reals = Vec::with_capacity(out.len());
        let mut imags = Vec::with_capacity(out.len());
        for c in out {
            match c {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!("Complex StaticArray cell expected"),
            }
        }
        return crate::helpers::static_array::base::build_static_array_from_flat_complex(b, reals, imags, new_shape);
    }
    build_static_array_from_flat(b, out, new_shape, dtype)
}

/// Try to apply `transpose` (or `.T`) natively. Returns `Some(result)` for
/// the migrated case; `None` otherwise.
pub fn try_apply_transpose(
    b: &mut IRBuilder,
    val: &Value,
    args: &[Value],
) -> Option<Value> {
    let dtype = match val {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => return None,
    };
    let _ = dtype;
    let shape = shape_of(val);
    let ndim = shape.len();
    if ndim <= 1 {
        return Some(val.clone());
    }
    let perm = parse_transpose_perm(args, ndim)?;
    let out = transpose_materialise(b, val, &perm);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    Some(out)
}

/// Try to apply `moveaxis` natively. Returns `Some(result)` for the migrated
/// case; `None` otherwise. `args` is `[source, dest]`.
pub fn try_apply_moveaxis(
    b: &mut IRBuilder,
    val: &Value,
    args: &[Value],
) -> Option<Value> {
    let dtype = match val {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => return None,
    };
    let _ = dtype;
    if args.len() < 2 {
        return None;
    }
    let src_raw = args[0].int_val()?;
    let dst_raw = args[1].int_val()?;
    let shape = shape_of(val);
    let ndim = shape.len();
    if ndim == 0 {
        return Some(val.clone());
    }
    let src = if src_raw < 0 { (ndim as i64 + src_raw) as usize } else { src_raw as usize };
    let dst = if dst_raw < 0 { (ndim as i64 + dst_raw) as usize } else { dst_raw as usize };
    if src >= ndim || dst >= ndim {
        panic!("moveaxis: axis out of bounds for array of rank {}", ndim);
    }
    // Build permutation: remove src, insert at dst.
    let mut perm: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
    perm.insert(dst, src);
    let out = transpose_materialise(b, val, &perm);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    Some(out)
}

// ────────────────────────────────────────────────────────────────────────
// expand_dims / squeeze / flatten / ravel
// ────────────────────────────────────────────────────────────────────────

/// Try to apply `expand_dims` natively. `args` is `[arr, axis]` where `axis`
/// is a static int.
pub fn try_apply_expand_dims(
    _b: &mut IRBuilder,
    val: &Value,
    axis_arg: Option<&Value>,
) -> Option<Value> {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => return None,
    };
    let axis_raw = axis_arg?.int_val()?;
    let new_ndim = shape.len() + 1;
    let resolved = if axis_raw < 0 { new_ndim as i64 + axis_raw } else { axis_raw };
    if resolved < 0 || resolved >= new_ndim as i64 {
        panic!(
            "expand_dims: axis {} is out of bounds for array of rank {}",
            axis_raw, new_ndim
        );
    }
    let pos = resolved as usize;
    let mut new_shape = shape.clone();
    new_shape.insert(pos, 1);
    let mut new_strides = strides.clone();
    // Inserted axis has length 1 → stride is irrelevant. Use 0 for clarity.
    new_strides.insert(pos, 0);
    let out = Value::StaticArray {
        dtype,
        shape: new_shape,
        segment_id,
        strides: new_strides,
        offset,
        imag_segment_id: imag_seg,
        value_id: ValueId::next(),
    };
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(_b, in_vid, out_vid);
    }
    Some(out)
}

/// Try to apply `squeeze` natively. `axis_arg` either drops a single given
/// length-1 dim, drops listed length-1 dims, or (when None) drops all
/// length-1 dims.
pub fn try_apply_squeeze(
    _b: &mut IRBuilder,
    val: &Value,
    axis_arg: Option<&Value>,
) -> Option<Value> {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => return None,
    };
    let ndim = shape.len();

    let target_axes: Vec<usize> = match axis_arg {
        None | Some(Value::None) => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
            .collect(),
        Some(Value::Tuple(d)) | Some(Value::List(d)) => {
            let mut out = Vec::with_capacity(d.values.len());
            for v in &d.values {
                let a = v.int_val()?;
                let r = if a < 0 { ndim as i64 + a } else { a };
                if r < 0 || r >= ndim as i64 {
                    panic!("squeeze: axis {} is out of bounds for array of rank {}", a, ndim);
                }
                out.push(r as usize);
            }
            out
        }
        Some(v) => {
            let a = v.int_val()?;
            let r = if a < 0 { ndim as i64 + a } else { a };
            if r < 0 || r >= ndim as i64 {
                panic!("squeeze: axis {} is out of bounds for array of rank {}", a, ndim);
            }
            vec![r as usize]
        }
    };
    for &ax in &target_axes {
        if shape[ax] != 1 {
            panic!(
                "squeeze: cannot select an axis to squeeze out which has size not equal to one (axis {})",
                ax
            );
        }
    }
    if target_axes.is_empty() {
        return Some(val.clone());
    }
    let mut new_shape: Vec<usize> = Vec::with_capacity(shape.len());
    let mut new_strides: Vec<usize> = Vec::with_capacity(strides.len());
    for (i, (&d, &s)) in shape.iter().zip(strides.iter()).enumerate() {
        if !target_axes.contains(&i) {
            new_shape.push(d);
            new_strides.push(s);
        }
    }
    let out = if new_shape.is_empty() {
        // 0-D scalar — read the single cell from the cache or segment.
        let cached = _b.static_array_payload.get(&segment_id)
            .and_then(|c| if offset < c.len() { Some(c[offset].clone()) } else { None });
        match cached {
            Some(v) => v,
            None => {
                if dtype == NumberType::Complex {
                    let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                    crate::helpers::static_array::read::read_complex_leaf(_b, segment_id, im, offset)
                } else {
                    let addr = _b.ir_constant_int(offset as i64);
                    let raw = _b.ir_read_memory(segment_id, &addr);
                    crate::ops::dyn_ndarray::scalar_i64_to_value(
                        &crate::ops::dyn_ndarray::value_to_scalar_i64(&raw),
                        dtype,
                    )
                }
            }
        }
    } else {
        Value::StaticArray {
            dtype,
            shape: new_shape,
            segment_id,
            strides: new_strides,
            offset,
            imag_segment_id: imag_seg,
            value_id: ValueId::next(),
        }
    };
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(_b, in_vid, out_vid);
    }
    Some(out)
}

/// Try to apply `flatten` / `ravel` natively. For contiguous source: pure
/// metadata. For non-contiguous: materialise.
pub fn try_apply_flatten(b: &mut IRBuilder, val: &Value) -> Option<Value> {
    let (dtype, shape, segment_id, _strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => return None,
    };
    let total: usize = shape.iter().product();
    let out = if is_contiguous(val) {
        Value::StaticArray {
            dtype,
            shape: vec![total],
            segment_id,
            strides: vec![1],
            offset,
            imag_segment_id: imag_seg,
            value_id: ValueId::next(),
        }
    } else if dtype == NumberType::Complex {
        // Non-contiguous: materialise into fresh contiguous segment.
        let cells = materialise_to_flat(b, val);
        let mut reals = Vec::with_capacity(cells.len());
        let mut imags = Vec::with_capacity(cells.len());
        for c in cells {
            match c {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!(),
            }
        }
        crate::helpers::static_array::base::build_static_array_from_flat_complex(b, reals, imags, vec![total])
    } else {
        let flat = materialise_to_flat(b, val);
        build_static_array_from_flat(b, flat, vec![total], dtype)
    };
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    Some(out)
}

// ────────────────────────────────────────────────────────────────────────
// concatenate / stack
// ────────────────────────────────────────────────────────────────────────

/// Promote any nested `Value::List` of pure-numeric leaves into a
/// `Value::StaticArray`. Used by concatenate/stack to handle inputs that are
/// still nested lists at the boundary (constant folding pipelines).
fn coerce_to_static_array(b: &mut IRBuilder, v: &Value) -> Option<Value> {
    match v {
        Value::StaticArray { .. } => Some(v.clone()),
        Value::List(_) | Value::Tuple(_) => super::base::to_static_array(b, v),
        _ => None,
    }
}

/// Try to apply `concatenate` natively. Returns `Some(result)` only when
/// every input is (or can be coerced to) a `Value::StaticArray` with
/// matching dtype and shapes (other than along the concatenation axis).
///
/// `arrays_arg` is the first positional argument (a list/tuple of arrays).
pub fn try_apply_concatenate(
    b: &mut IRBuilder,
    arrays_arg: &Value,
    axis: i64,
) -> Option<Value> {
    let inputs = match arrays_arg {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if inputs.is_empty() {
        return None;
    }
    let mut arrays: Vec<Value> = Vec::with_capacity(inputs.len());
    for v in &inputs {
        let arr = coerce_to_static_array(b, v)?;
        arrays.push(arr);
    }
    let has_complex = arrays.iter().any(|a| matches!(a, Value::StaticArray { dtype: NumberType::Complex, .. }));
    let all_complex = arrays.iter().all(|a| matches!(a, Value::StaticArray { dtype: NumberType::Complex, .. }));
    if has_complex && !all_complex {
        return None;
    }

    let first_shape = shape_of(&arrays[0]);
    let ndim = first_shape.len();
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "concatenate: axis {} is out of bounds for array with {} dimensions",
            axis, ndim
        );
    }
    let ax = resolved as usize;

    // Validate shapes match in all axes except `ax`.
    for arr in &arrays[1..] {
        let s = shape_of(arr);
        if s.len() != ndim {
            panic!("concatenate: all input arrays must have the same number of dimensions");
        }
        for (i, (&a, &b)) in first_shape.iter().zip(s.iter()).enumerate() {
            if i != ax && a != b {
                panic!(
                    "concatenate: all input array dimensions except for the concatenation axis must match exactly"
                );
            }
        }
    }

    // Promote dtype: any Float input → Float output.
    let mut out_dtype = NumberType::Integer;
    if all_complex {
        out_dtype = NumberType::Complex;
    } else {
        for arr in &arrays {
            if matches!(dtype_of(arr), NumberType::Float) {
                out_dtype = NumberType::Float;
                break;
            }
        }
    }

    // Compute output shape.
    let total_axis: usize = arrays.iter().map(|a| shape_of(a)[ax]).sum();
    let mut out_shape = first_shape.clone();
    out_shape[ax] = total_axis;
    let out_total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    // Read each input as flat row-major.
    let inputs_flat: Vec<Vec<Value>> = arrays.iter().map(|a| materialise_to_flat(b, a)).collect();
    let inputs_strides: Vec<Vec<usize>> = arrays.iter()
        .map(|a| row_major_strides(&shape_of(a)))
        .collect();
    let inputs_axis_lens: Vec<usize> = arrays.iter().map(|a| shape_of(a)[ax]).collect();
    // Offsets along `ax` where each input starts.
    let mut axis_starts: Vec<usize> = Vec::with_capacity(arrays.len());
    {
        let mut acc = 0;
        for &l in &inputs_axis_lens {
            axis_starts.push(acc);
            acc += l;
        }
    }

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for flat_out in 0..out_total {
        let coords = decode_coords(flat_out, &out_shape, &out_strides);
        // Find which input owns this output coord along axis ax.
        let coord_ax = coords[ax];
        let mut input_idx = 0usize;
        for (i, &l) in inputs_axis_lens.iter().enumerate() {
            if coord_ax < axis_starts[i] + l {
                input_idx = i;
                break;
            }
        }
        let local_ax = coord_ax - axis_starts[input_idx];
        // Build source coords for the chosen input.
        let mut src_lin: usize = 0;
        for k in 0..ndim {
            let c = if k == ax { local_ax } else { coords[k] };
            src_lin += c * inputs_strides[input_idx][k];
        }
        let cell = inputs_flat[input_idx][src_lin].clone();
        out_flat.push(promote_cell_dtype(b, &cell, out_dtype));
    }
    if out_dtype == NumberType::Complex {
        let mut reals = Vec::with_capacity(out_flat.len());
        let mut imags = Vec::with_capacity(out_flat.len());
        for c in out_flat {
            match c {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!("Complex concat: cell must be Complex"),
            }
        }
        return Some(crate::helpers::static_array::base::build_static_array_from_flat_complex(b, reals, imags, out_shape));
    }
    Some(build_static_array_from_flat(b, out_flat, out_shape, out_dtype))
}

/// Promote a leaf to the target dtype (Integer→Float when needed).
fn promote_cell_dtype(b: &mut IRBuilder, v: &Value, target: NumberType) -> Value {
    match (target, v) {
        (NumberType::Float, Value::Float(_)) => v.clone(),
        (NumberType::Float, _) => b.ir_float_cast(v),
        _ => v.clone(),
    }
}

/// Try to apply `stack` natively (insert a new axis at position `axis` of
/// length `len(arrays)`).
pub fn try_apply_stack(
    b: &mut IRBuilder,
    arrays_arg: &Value,
    axis: i64,
) -> Option<Value> {
    let inputs = match arrays_arg {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if inputs.is_empty() {
        return None;
    }
    let mut arrays: Vec<Value> = Vec::with_capacity(inputs.len());
    for v in &inputs {
        let arr = coerce_to_static_array(b, v)?;
        arrays.push(arr);
    }
    let has_complex = arrays.iter().any(|a| matches!(a, Value::StaticArray { dtype: NumberType::Complex, .. }));
    let all_complex = arrays.iter().all(|a| matches!(a, Value::StaticArray { dtype: NumberType::Complex, .. }));
    if has_complex && !all_complex { return None; }

    let first_shape = shape_of(&arrays[0]);
    let ndim = first_shape.len();
    let new_ndim = ndim + 1;
    let resolved = if axis < 0 { new_ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= new_ndim as i64 {
        panic!("stack: axis {} is out of bounds for array of dimension {}", axis, ndim);
    }
    let ax = resolved as usize;

    // Validate all input shapes equal.
    for arr in &arrays[1..] {
        if shape_of(arr) != first_shape {
            panic!("stack: all input arrays must have the same shape");
        }
    }

    // Promote dtype.
    let mut out_dtype = NumberType::Integer;
    if all_complex {
        out_dtype = NumberType::Complex;
    } else {
        for arr in &arrays {
            if matches!(dtype_of(arr), NumberType::Float) {
                out_dtype = NumberType::Float;
                break;
            }
        }
    }

    // Compute output shape (insert len(arrays) at position `ax`).
    let mut out_shape: Vec<usize> = first_shape.clone();
    out_shape.insert(ax, arrays.len());
    let out_total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    let inputs_flat: Vec<Vec<Value>> = arrays.iter().map(|a| materialise_to_flat(b, a)).collect();
    let in_strides = row_major_strides(&first_shape);

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for flat_out in 0..out_total {
        let coords = decode_coords(flat_out, &out_shape, &out_strides);
        let input_idx = coords[ax];
        // Source coords are out_coords minus the inserted axis.
        let mut src_lin: usize = 0;
        let mut k = 0;
        for d in 0..new_ndim {
            if d == ax { continue; }
            src_lin += coords[d] * in_strides[k];
            k += 1;
        }
        let cell = inputs_flat[input_idx][src_lin].clone();
        out_flat.push(promote_cell_dtype(b, &cell, out_dtype));
    }
    if out_dtype == NumberType::Complex {
        let mut reals = Vec::with_capacity(out_flat.len());
        let mut imags = Vec::with_capacity(out_flat.len());
        for c in out_flat {
            match c {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!("Complex stack: cell must be Complex"),
            }
        }
        return Some(crate::helpers::static_array::base::build_static_array_from_flat_complex(b, reals, imags, out_shape));
    }
    Some(build_static_array_from_flat(b, out_flat, out_shape, out_dtype))
}

// ────────────────────────────────────────────────────────────────────────
// Stack convenience wrappers (vstack / hstack / column_stack)
// ────────────────────────────────────────────────────────────────────────

/// Promote a 1-D StaticArray to a 2-D row (`(N,)` → `(1, N)`); higher-rank
/// arrays pass through. Always returns a StaticArray.
fn promote_to_row(b: &mut IRBuilder, v: &Value) -> Option<Value> {
    let arr = coerce_to_static_array(b, v)?;
    let s = shape_of(&arr);
    if s.len() < 2 {
        // expand_dims at axis=0.
        let zero = b.ir_constant_int(0);
        try_apply_expand_dims(b, &arr, Some(&zero))
    } else {
        Some(arr)
    }
}

/// Promote a 1-D StaticArray to a 2-D column (`(N,)` → `(N, 1)`); higher-rank
/// arrays pass through.
fn promote_to_column(b: &mut IRBuilder, v: &Value) -> Option<Value> {
    let arr = coerce_to_static_array(b, v)?;
    let s = shape_of(&arr);
    if s.len() == 1 {
        let one = b.ir_constant_int(1);
        try_apply_expand_dims(b, &arr, Some(&one))
    } else {
        Some(arr)
    }
}

/// Try to apply `vstack` natively.
pub fn try_apply_vstack(b: &mut IRBuilder, arrays_arg: &Value) -> Option<Value> {
    let inputs = match arrays_arg {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if inputs.is_empty() {
        return None;
    }
    let mut promoted: Vec<Value> = Vec::with_capacity(inputs.len());
    for v in &inputs {
        let p = promote_to_row(b, v)?;
        promoted.push(p);
    }
    let promoted_list = Value::List(crate::types::CompositeData {
        elements_type: promoted.iter().map(|v| v.zinnia_type()).collect(),
        values: promoted,
    
        value_id: ValueId::next(),
    });
    try_apply_concatenate(b, &promoted_list, 0)
}

/// Try to apply `hstack` natively. NumPy convention: 1-D inputs concat along
/// axis 0; ≥2-D inputs concat along axis 1.
pub fn try_apply_hstack(b: &mut IRBuilder, arrays_arg: &Value) -> Option<Value> {
    let inputs = match arrays_arg {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if inputs.is_empty() {
        return None;
    }
    // Coerce each to StaticArray first to determine ranks.
    let mut arrays: Vec<Value> = Vec::with_capacity(inputs.len());
    for v in &inputs {
        let arr = coerce_to_static_array(b, v)?;
        arrays.push(arr);
    }
    let any_multi = arrays.iter().any(|a| shape_of(a).len() >= 2);
    let axis = if any_multi { 1i64 } else { 0i64 };
    let arrays_list = Value::List(crate::types::CompositeData {
        elements_type: arrays.iter().map(|v| v.zinnia_type()).collect(),
        values: arrays,
    
        value_id: ValueId::next(),
    });
    try_apply_concatenate(b, &arrays_list, axis)
}

/// Try to apply `column_stack` natively.
pub fn try_apply_column_stack(b: &mut IRBuilder, arrays_arg: &Value) -> Option<Value> {
    let inputs = match arrays_arg {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => return None,
    };
    if inputs.is_empty() {
        return None;
    }
    let mut promoted: Vec<Value> = Vec::with_capacity(inputs.len());
    for v in &inputs {
        let p = promote_to_column(b, v)?;
        promoted.push(p);
    }
    let promoted_list = Value::List(crate::types::CompositeData {
        elements_type: promoted.iter().map(|v| v.zinnia_type()).collect(),
        values: promoted,
    
        value_id: ValueId::next(),
    });
    try_apply_concatenate(b, &promoted_list, 1)
}

// Marker used in tests; silences unused HashMap import for the time being.
#[allow(dead_code)]
fn _unused_hashmap_marker() -> HashMap<String, Value> {
    HashMap::new()
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
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    fn make_2d_int(b: &mut IRBuilder, rows: &[&[i64]]) -> Value {
        let row_lists: Vec<Value> = rows
            .iter()
            .map(|r| list_of(r.iter().map(|n| b.ir_constant_int(*n)).collect()))
            .collect();
        let lst = list_of(row_lists);
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    #[test]
    fn reshape_2x3_to_3x2_metadata_view() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let three = b.ir_constant_int(3);
        let two = b.ir_constant_int(2);
        let out = try_apply_reshape(&mut b, &a, &[three, two]).expect("native");
        match (&a, &out) {
            (
                Value::StaticArray { segment_id: a_seg, .. },
                Value::StaticArray { segment_id: o_seg, shape, strides, .. },
            ) => {
                assert_eq!(a_seg, o_seg, "reshape on contiguous source must reuse segment");
                assert_eq!(*shape, vec![3, 2]);
                assert_eq!(*strides, vec![2, 1]);
            }
            _ => panic!("expected StaticArray"),
        }
    }

    #[test]
    fn reshape_with_neg_one_inference() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let neg_one = b.ir_constant_int(-1);
        let out = try_apply_reshape(&mut b, &a, &[neg_one]).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![6]),
            _ => panic!(),
        }
        let two = b.ir_constant_int(2);
        let neg_one2 = b.ir_constant_int(-1);
        let out2 = try_apply_reshape(&mut b, &a, &[two, neg_one2]).expect("native");
        match &out2 {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2, 3]),
            _ => panic!(),
        }
    }

    #[test]
    fn transpose_materialises_with_constant_fold() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let out = try_apply_transpose(&mut b, &a, &[]).expect("native");
        match &out {
            Value::StaticArray { shape, strides, .. } => {
                assert_eq!(*shape, vec![3, 2]);
                assert_eq!(*strides, vec![2, 1]);
            }
            _ => panic!(),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        // Transposed: [[1,4],[2,5],[3,6]] flat → [1, 4, 2, 5, 3, 6]
        assert_eq!(v, vec![Some(1), Some(4), Some(2), Some(5), Some(3), Some(6)]);
    }

    #[test]
    fn concatenate_axis0_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2]);
        let c = make_1d_int(&mut b, &[3, 4]);
        let arrays_list = list_of(vec![a, c]);
        let out = try_apply_concatenate(&mut b, &arrays_list, 0).expect("native");
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(1), Some(2), Some(3), Some(4)]);
    }

    #[test]
    fn concatenate_2d_axis0() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2], &[3, 4]]);
        let c = make_2d_int(&mut b, &[&[5, 6], &[7, 8]]);
        let arrays_list = list_of(vec![a, c]);
        let out = try_apply_concatenate(&mut b, &arrays_list, 0).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![4, 2]),
            _ => panic!(),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(1), Some(2), Some(3), Some(4),
                            Some(5), Some(6), Some(7), Some(8)]);
    }

    #[test]
    fn concatenate_2d_axis1() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2], &[3, 4]]);
        let c = make_2d_int(&mut b, &[&[5, 6], &[7, 8]]);
        let arrays_list = list_of(vec![a, c]);
        let out = try_apply_concatenate(&mut b, &arrays_list, 1).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2, 4]),
            _ => panic!(),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(1), Some(2), Some(5), Some(6),
                            Some(3), Some(4), Some(7), Some(8)]);
    }

    #[test]
    fn stack_axis0_inserts_new_axis() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let c = make_1d_int(&mut b, &[4, 5, 6]);
        let arrays_list = list_of(vec![a, c]);
        let out = try_apply_stack(&mut b, &arrays_list, 0).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2, 3]),
            _ => panic!(),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]);
    }

    #[test]
    fn expand_dims_metadata() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let zero = b.ir_constant_int(0);
        let out = try_apply_expand_dims(&mut b, &a, Some(&zero)).expect("native");
        match (&a, &out) {
            (
                Value::StaticArray { segment_id: a_seg, .. },
                Value::StaticArray { segment_id: o_seg, shape, .. },
            ) => {
                assert_eq!(a_seg, o_seg);
                assert_eq!(*shape, vec![1, 3]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn squeeze_drops_length_one_dims() {
        let mut b = IRBuilder::new();
        // Build a (1, 3, 1) StaticArray by expand_dims twice.
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let zero = b.ir_constant_int(0);
        let inserted0 = try_apply_expand_dims(&mut b, &a, Some(&zero)).unwrap();
        let two = b.ir_constant_int(2);
        let inserted1 = try_apply_expand_dims(&mut b, &inserted0, Some(&two)).unwrap();
        let out = try_apply_squeeze(&mut b, &inserted1, None).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3]),
            _ => panic!(),
        }
    }

    #[test]
    fn flatten_metadata_on_contiguous() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let out = try_apply_flatten(&mut b, &a).expect("native");
        match (&a, &out) {
            (
                Value::StaticArray { segment_id: a_seg, .. },
                Value::StaticArray { segment_id: o_seg, shape, strides, .. },
            ) => {
                assert_eq!(a_seg, o_seg);
                assert_eq!(*shape, vec![6]);
                assert_eq!(*strides, vec![1]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn moveaxis_basic() {
        let mut b = IRBuilder::new();
        // shape (2, 3, 4): axis 0 → axis 2 results in (3, 4, 2)
        let mut leaves = Vec::new();
        for v in 0..24 { leaves.push(b.ir_constant_int(v)); }
        let lst = list_of(leaves);
        let arr = super::super::base::build_static_array_from_flat(
            &mut b, match &lst { Value::List(d) => d.values.clone(), _ => unreachable!() },
            vec![2, 3, 4], NumberType::Integer,
        );
        let zero = b.ir_constant_int(0);
        let two = b.ir_constant_int(2);
        let out = try_apply_moveaxis(&mut b, &arr, &[zero, two]).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3, 4, 2]),
            _ => panic!(),
        }
    }
}
