use crate::builder::IRBuilder;
use super::value_to_scalar_i64;
use crate::helpers::shape_arith::row_major_strides;
use crate::optim::resolver::{require_provable_static_int, SiteKind};
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, ScalarValue, Value, ValueId, ZinniaType,
};

/// Transpose a dynamic array as a **lazy stride-permuted view**.
///
/// Shares the source segment; permutes `logical_shape` / `logical_strides`
/// / `runtime_shape` / `runtime_strides`; preserves `logical_offset`. No
/// `ir_write_memory` is emitted — the materialisation cost is deferred to
/// consumers that need contiguous layout (which call
/// [`super::force_contiguous`]).
///
/// Op chains like `slice → reshape → swapaxes → reshape` benefit most:
/// previously each link in the chain materialised the full array (4×
/// `ir_write_memory` storm for 0388-style kernels). Now only consumers
/// that actually need contiguous data pay materialisation.
pub fn dyn_transpose(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
    let ndim = shape.len();

    if ndim <= 1 {
        return Value::DynamicNDArray(data.clone());
    }

    // Determine axis permutation
    let perm: Vec<usize> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
        (0..ndim).rev().collect()
    } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) = args.first() {
        perm_data
            .values
            .iter()
            .map(|v| {
                let a: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                let resolved = if a < 0 { ndim as i64 + a } else { a };
                resolved as usize
            })
            .collect()
    } else {
        args.iter()
            .map(|v| {
                let a: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                let resolved = if a < 0 { ndim as i64 + a } else { a };
                resolved as usize
            })
            .collect()
    };

    assert_eq!(
        perm.len(),
        ndim,
        "transpose: permutation length must match rank"
    );

    // A runtime-strided source (runtime-reshape output: compact buffer in
    // runtime C-order) cannot be transposed as a static stride
    // permutation — the skeleton strides describe the padding rectangle,
    // not the content layout. Materialise a permuted copy with
    // runtime-stride address arithmetic instead (case 0389: swapaxes
    // after `reshape(h//r, -1, r, c)` was silently the identity).
    //
    // A padded buffer (elements at their row-major position in the
    // envelope: a bounded input, a runtime-bounds slice) is a plain stride
    // permutation even when its extents are runtime values
    // (`has_padded_layout`); a bounded input's recorded `runtime_strides`
    // are C-order ones and must not be used to address it (`a.T.sum()` of
    // a 2x2 input in a 4x3 envelope proved 0).
    let runtime_extents = data.meta.runtime_shape.iter().any(|sv| sv.static_val.is_none());
    if runtime_extents && !super::has_padded_layout(data) {
        return dyn_transpose_runtime_strided(b, data, &perm);
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();

    // Permute envelope dims. DimVars preserved — unifications stay valid.
    let new_dims: Vec<crate::types::Dim> = perm.iter().map(|&p| data.envelope.dims[p]).collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, data.envelope.total_bound);

    let new_runtime_shape: Vec<ScalarValue<i64>> = perm
        .iter()
        .map(|&p| {
            if p < data.meta.runtime_shape.len() {
                data.meta.runtime_shape[p].clone()
            } else {
                ScalarValue::new(Some(shape[p] as i64), None)
            }
        })
        .collect();

    // The view's addressing is the permuted buffer strides.
    let new_runtime_strides: Vec<ScalarValue<i64>> = new_strides
        .iter()
        .map(|&s| ScalarValue::new(Some(s as i64), None))
        .collect();

    let view = DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id, // ← reuse: this is the lazy-view win
        meta: DynArrayMeta {
            logical_shape: new_shape,
            logical_offset: data.meta.logical_offset,
            logical_strides: new_strides,
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: data.meta.runtime_rank.clone(),
            runtime_shape: new_runtime_shape,
            runtime_strides: new_runtime_strides,
            runtime_offset: data.meta.runtime_offset.clone(),
        },
        value_id: ValueId::next(),
    };
    // With runtime extents the result is materialised (a padded copy in
    // the permuted envelope): row reads and the output layout of a
    // runtime-shaped array assume a row-major buffer (`a.T[0]` of a padded
    // input read a row of `a`, and `return a.T` exposed `a`).
    if runtime_extents {
        return Value::DynamicNDArray(super::force_contiguous(b, &view));
    }
    Value::DynamicNDArray(view)
}

/// Transpose a runtime-strided dyn array by materialising a permuted
/// copy. Each storage position `s` of the compact buffer decodes to
/// runtime coordinates (mixed-radix over the runtime dims), which map to
/// an output position under the permuted runtime C-order strides; the
/// cell scatters there with a dynamic write. Positions beyond
/// `runtime_length` self-map (identity) so padding stays padding.
fn dyn_transpose_runtime_strided(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    perm: &[usize],
) -> Value {
    let rank = perm.len();
    let total = data.envelope.total_bound;
    let shape = &data.meta.logical_shape;

    // Runtime dims as IR values.
    let dim_vals: Vec<Value> = data
        .meta
        .runtime_shape
        .iter()
        .map(|sv| {
            if let Some(c) = sv.static_val {
                b.ir_constant_int(c)
            } else {
                Value::Integer(sv.clone())
            }
        })
        .collect();

    // Output runtime strides: C-order over the permuted dims.
    let mut out_strides: Vec<Value> = vec![b.ir_constant_int(1); rank];
    let mut acc = b.ir_constant_int(1);
    for i in (0..rank).rev() {
        out_strides[i] = acc.clone();
        if i > 0 {
            acc = b.ir_mul_i(&acc, &dim_vals[perm[i]]);
        }
    }
    // out axis position of each SOURCE axis.
    let mut out_axis_of_src: Vec<usize> = vec![0; rank];
    for (out_ax, &src_ax) in perm.iter().enumerate() {
        out_axis_of_src[src_ax] = out_ax;
    }

    let runtime_len_v: Value = if let Some(c) = data.meta.runtime_length.static_val {
        b.ir_constant_int(c)
    } else {
        Value::Integer(data.meta.runtime_length.clone())
    };
    let len_static_full = data.meta.runtime_length.static_val == Some(total as i64);

    let default_val = super::metadata::dyn_default_value(b, data.dtype);
    // A complex element is two cells (`helpers::segment`).
    let cells = if data.dtype == crate::types::NumberType::Complex { 2 * total } else { total };
    let zero = b.ir_constant_int(0);
    let out_seg = b.alloc_filled(cells, &zero, data.dtype);

    // Walk storage positions with an incremental mixed-radix carry chain
    // (eq/select/add only — a per-cell div/mod decode fires the
    // floor_div/mod op contracts' Z3 discharge 2·rank·total times, which
    // is what made 20k-cell kernels uncompilable).
    let zero_c = b.ir_constant_int(0);
    let one_c = b.ir_constant_int(1);
    let mut coords: Vec<Value> = vec![zero_c.clone(); rank];
    for s in 0..total {
        let s_c = b.ir_constant_int(s as i64);
        // Output position under permuted strides.
        let mut out_pos = b.ir_constant_int(0);
        for src_ax in 0..rank {
            let contrib = b.ir_mul_i(&coords[src_ax], &out_strides[out_axis_of_src[src_ax]]);
            out_pos = b.ir_add_i(&out_pos, &contrib);
        }
        let cell = crate::helpers::segment::read_typed(b, data.segment_id, &s_c, data.dtype);
        if len_static_full {
            crate::helpers::segment::write_typed(b, out_seg, &out_pos, &cell, data.dtype);
        } else {
            // Inactive tail: keep the padding at its own position with the
            // default value so active scatters never collide with it.
            let active = b.ir_less_than_i(&s_c, &runtime_len_v);
            let addr = b.ir_select_i(&active, &out_pos, &s_c);
            let val = if data.dtype == crate::types::NumberType::Float {
                b.ir_select_f(&active, &cell, &default_val)
            } else {
                b.ir_select_i(&active, &cell, &default_val)
            };
            crate::helpers::segment::write_typed(b, out_seg, &addr, &val, data.dtype);
        }
        // Increment the runtime coordinates (last axis fastest).
        if s + 1 < total {
            let mut carry = one_c.clone();
            for ax in (0..rank).rev() {
                let bumped = b.ir_add_i(&coords[ax], &carry);
                if ax == 0 {
                    coords[0] = bumped;
                    break;
                }
                let wrap = b.ir_equal_i(&bumped, &dim_vals[ax]);
                coords[ax] = b.ir_select_i(&wrap, &zero_c, &bumped);
                carry = b.ir_select_i(&wrap, &one_c, &zero_c);
            }
        }
    }

    let new_logical: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_log_strides = row_major_strides(&new_logical);
    let new_dims: Vec<crate::types::Dim> = perm.iter().map(|&p| data.envelope.dims[p]).collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, total);
    let new_runtime_shape: Vec<ScalarValue<i64>> = perm
        .iter()
        .map(|&p| data.meta.runtime_shape[p].clone())
        .collect();
    let new_runtime_strides: Vec<ScalarValue<i64>> = out_strides
        .iter()
        .map(crate::ops::dyn_ndarray::value_to_scalar_i64)
        .collect();

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: out_seg,
        meta: DynArrayMeta {
            logical_shape: new_logical,
            logical_offset: 0,
            logical_strides: new_log_strides,
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: data.meta.runtime_rank.clone(),
            runtime_shape: new_runtime_shape,
            runtime_strides: new_runtime_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

pub fn dyn_moveaxis(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let ndim = data.meta.logical_shape.len();
    assert!(args.len() >= 2, "moveaxis: requires source and destination");

    let src = {
        let s: i64 = require_provable_static_int(b, &args[0], SiteKind::Axis);
        if s < 0 {
            (ndim as i64 + s) as usize
        } else {
            s as usize
        }
    };
    let dst = {
        let d: i64 = require_provable_static_int(b, &args[1], SiteKind::Axis);
        if d < 0 {
            (ndim as i64 + d) as usize
        } else {
            d as usize
        }
    };
    assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

    // Build permutation: remove src, insert at dst
    let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
    order.insert(dst, src);

    let axes_val: Vec<Value> = order
        .iter()
        .map(|&a| Value::Integer(ScalarValue::new(Some(a as i64), None)))
        .collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; order.len()],
        values: axes_val,

        value_id: ValueId::next(),
    });
    dyn_transpose(b, data, &[axes_tuple])
}

pub fn dyn_swapaxes(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let ndim = data.meta.logical_shape.len();
    assert!(args.len() >= 2, "swapaxes: requires axis1 and axis2");
    let axis1 = normalize_axis(
        require_provable_static_int(b, &args[0], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let axis2 = normalize_axis(
        require_provable_static_int(b, &args[1], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let mut order: Vec<usize> = (0..ndim).collect();
    order.swap(axis1, axis2);
    let axes_val: Vec<Value> = order
        .iter()
        .map(|&a| Value::Integer(ScalarValue::new(Some(a as i64), None)))
        .collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; order.len()],
        values: axes_val,
        value_id: ValueId::next(),
    });
    dyn_transpose(b, data, &[axes_tuple])
}

/// Insert a size-1 axis (np.expand_dims) as a metadata-only change on a
/// contiguous buffer.
pub fn dyn_expand_dims(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let ndim = data.meta.logical_shape.len();
    let axis_i: i64 = require_provable_static_int(
        b,
        args.first().expect("expand_dims: axis required"),
        SiteKind::Axis,
    );
    let axis = if axis_i < 0 {
        (ndim as i64 + 1 + axis_i) as usize
    } else {
        axis_i as usize
    };
    assert!(
        axis <= ndim,
        "expand_dims: axis {} out of bounds for rank {}",
        axis_i,
        ndim
    );
    let data = super::force_contiguous(b, data);
    let mut shape = data.meta.logical_shape.clone();
    shape.insert(axis, 1);
    let strides = row_major_strides(&shape);
    let mut dims = data.envelope.dims.clone();
    dims.insert(axis, crate::types::Dim::new_static(&mut b.dim_table, 1));
    let envelope = crate::types::Envelope::new_with_bound(dims, data.envelope.total_bound);
    let mut rt_shape = data.meta.runtime_shape.clone();
    rt_shape.insert(axis, ScalarValue::new(Some(1), None));
    // The size-1 axis is never multiplied by a nonzero coordinate, so any
    // stride works; reuse the stride of the axis that now follows (or 1).
    let mut rt_strides = data.meta.runtime_strides.clone();
    let ins = rt_strides
        .get(axis)
        .cloned()
        .unwrap_or_else(|| ScalarValue::new(Some(1), None));
    rt_strides.insert(axis, ins);
    let runtime_rank = ScalarValue::new(
        data.meta.runtime_rank.static_val.map(|r| r + 1),
        data.meta.runtime_rank.stmt_id,
    );

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id,
        meta: DynArrayMeta {
            logical_shape: shape,
            logical_offset: data.meta.logical_offset,
            logical_strides: strides,
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank,
            runtime_shape: rt_shape,
            runtime_strides: rt_strides,
            runtime_offset: data.meta.runtime_offset.clone(),
        },
        value_id: ValueId::next(),
    })
}

pub fn dyn_squeeze(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let ndim = data.meta.logical_shape.len();
    let axes: Vec<usize> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
        data.meta
            .logical_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| if dim == 1 { Some(i) } else { None })
            .collect()
    } else {
        let axis_vals: Vec<&Value> = match &args[0] {
            Value::Tuple(d) | Value::List(d) => d.values.iter().collect(),
            v => vec![v],
        };
        axis_vals
            .iter()
            .map(|v| {
                normalize_axis(
                    require_provable_static_int(b, v, SiteKind::Axis),
                    ndim,
                    "squeeze",
                )
            })
            .collect()
    };

    for &axis in &axes {
        assert!(
            data.meta.logical_shape[axis] == 1,
            "squeeze: cannot squeeze axis {} with max dimension {} on bounded DynamicNDArray",
            axis,
            data.meta.logical_shape[axis],
        );
    }

    let mut new_shape = Vec::new();
    let mut new_runtime_shape = Vec::new();
    for (axis, &dim) in data.meta.logical_shape.iter().enumerate() {
        if axes.contains(&axis) {
            continue;
        }
        new_shape.push(dim);
        new_runtime_shape.push(data.meta.runtime_shape[axis].clone());
    }
    let new_strides = row_major_strides(&new_shape);
    let new_dims: Vec<crate::types::Dim> = new_shape
        .iter()
        .map(|&s| crate::types::Dim::new_static(&mut b.dim_table, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, data.envelope.total_bound);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id,
        meta: DynArrayMeta {
            logical_shape: new_shape.clone(),
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: ScalarValue::new(Some(new_shape.len() as i64), None),
            runtime_shape: new_runtime_shape,
            runtime_strides: new_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

fn normalize_axis(axis: i64, ndim: usize, op: &str) -> usize {
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    assert!(
        resolved >= 0 && (resolved as usize) < ndim,
        "{}: axis {} out of bounds for rank {}",
        op,
        axis,
        ndim
    );
    resolved as usize
}

/// Reshape with one or more runtime-int shape elements. The source must
/// have a statically-known total element count (used as `total_bound`).
/// At runtime, the product of the supplied dim values must equal the
/// source total — enforced via a circuit assertion.
///
/// `logical_shape` is over-approximated: each runtime dim is recorded as
/// `old_total` (the loosest upper bound implied by the source). Downstream
/// ops that need true per-dim bounds should consult `runtime_shape`.
/// Padded-prefix strided view: when a runtime reshape's source is a
/// PADDED buffer (a dynamic-bounds slice of a static array: static-full
/// leading axes, one partial trailing axis, envelope strides) and the
/// target keeps the static prefix verbatim, the reshape must NOT
/// address the segment with compact strides — each prefix block lives
/// at its envelope offset. Returns the strided-view metadata
/// (runtime_strides, runtime_length, block-confinement record) over the
/// SAME segment; `None` means the source is compact and the standard
/// compact-stride path is correct.
///
/// Soundness of the confinement record: the caller's
/// `prod(runtime_shape) == runtime_length` assert forces the tail dims'
/// product to equal the source's trailing extent, and the extent is
/// `<= envelope dim` by slice-bound construction, so every active
/// element's window stays inside its prefix block.
fn padded_prefix_view(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    shape_values: &[Value],
) -> Option<(
    Vec<ScalarValue<i64>>,
    ScalarValue<i64>,
    crate::provenance::BlockView,
)> {
    let meta = &data.meta;
    if meta.runtime_offset.static_val != Some(0) {
        return None;
    }
    let src_rank = meta.logical_shape.len();
    if src_rank < 2 || meta.runtime_shape.len() != src_rank {
        return None;
    }
    // Source must address through the static envelope strides.
    let env_strides = row_major_strides(&meta.logical_shape);
    if meta.runtime_strides.len() != src_rank {
        return None;
    }
    for (sv, &e) in meta.runtime_strides.iter().zip(env_strides.iter()) {
        if sv.static_val != Some(e as i64) {
            return None;
        }
    }
    // Leading axes static-full; trailing axis genuinely partial.
    let prefix_rank = src_rank - 1;
    for i in 0..prefix_rank {
        if meta.runtime_shape[i].static_val != Some(meta.logical_shape[i] as i64) {
            return None;
        }
    }
    if meta.runtime_shape[src_rank - 1].static_val
        == Some(meta.logical_shape[src_rank - 1] as i64)
    {
        return None; // full buffer — compact path is correct and cheaper
    }
    // Target must preserve the static prefix verbatim and have a tail.
    if shape_values.len() <= prefix_rank {
        return None;
    }
    for i in 0..prefix_rank {
        if shape_values[i].int_val() != Some(meta.logical_shape[i] as i64) {
            return None;
        }
    }
    // Strides: prefix axes keep their envelope strides; tail axes get
    // row-major strides over the (possibly dynamic) tail dims.
    let n = shape_values.len();
    let mut rt: Vec<ScalarValue<i64>> = vec![ScalarValue::new(Some(1), None); n];
    let mut acc = b.ir_constant_int(1);
    for i in (prefix_rank..n).rev() {
        rt[i] = value_to_scalar_i64(&acc);
        if i > prefix_rank {
            acc = b.ir_mul_i(&acc, &shape_values[i]);
        }
    }
    for (i, slot) in rt.iter_mut().enumerate().take(prefix_rank) {
        *slot = ScalarValue::new(Some(env_strides[i] as i64), None);
    }
    Some((
        rt,
        meta.runtime_length.clone(),
        crate::provenance::BlockView {
            prefix_rank,
            block: env_strides[prefix_rank - 1],
        },
    ))
}

fn dyn_reshape_runtime(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    dim_args: &[&Value],
    old_total: usize,
) -> Value {
    let n_dims = dim_args.len();

    // Build runtime_shape (carries the actual values, possibly with ptr) and
    // logical_shape (compile-time skeleton). The KEY INVARIANT we maintain:
    //   prod(logical_shape) == total_bound
    // so downstream ops that consult logical_shape (e.g. allocate / iterate
    // over slice positions) don't over-count beyond the source storage.
    //
    // For each runtime dim we pick logical size 1, then place the entire
    // total_bound in the first runtime axis. (If all dims are constants their
    // product already equals total_bound by the assert below.)
    let mut runtime_shape: Vec<ScalarValue<i64>> = Vec::with_capacity(n_dims);
    let mut shape_values: Vec<Value> = Vec::with_capacity(n_dims);
    let mut logical_shape: Vec<usize> = Vec::with_capacity(n_dims);
    let mut const_product: usize = 1;
    let mut runtime_axes: Vec<usize> = Vec::new();
    for (axis, arg) in dim_args.iter().enumerate() {
        match arg.int_val() {
            Some(c) => {
                assert!(c > 0, "reshape: dimensions must be positive");
                runtime_shape.push(ScalarValue::new(Some(c), None));
                shape_values.push(b.ir_constant_int(c));
                logical_shape.push(c as usize);
                const_product = const_product
                    .checked_mul(c as usize)
                    .expect("reshape: const dim product overflow");
            }
            None => {
                let scalar = match arg {
                    Value::Integer(s) => s.clone(),
                    other => match b.ir_int_cast(other) {
                        Value::Integer(s) => s,
                        _ => unreachable!("ir_int_cast returns Value::Integer"),
                    },
                };
                runtime_shape.push(scalar.clone());
                shape_values.push(Value::Integer(scalar));
                // Tentatively 1; we'll redistribute total_bound to the first
                // runtime axis below.
                logical_shape.push(1);
                runtime_axes.push(axis);
            }
        }
    }
    if !runtime_axes.is_empty() {
        assert!(
            const_product > 0 && old_total % const_product == 0,
            "reshape: const dims ({:?}) don't divide source total {}",
            const_product,
            old_total,
        );
        // Place all the runtime variability into the first runtime axis so
        // prod(logical_shape) == old_total.
        logical_shape[runtime_axes[0]] = old_total / const_product;
    }

    // Assert prod(runtime_shape) == array's runtime length at proof time.
    // Use the array's runtime_length (slice-aware), not the parent's static
    // envelope bound, so reshapes on sliced arrays verify against the actual
    // slice length rather than the original full size.
    let length_v = match data.meta.runtime_length.stmt_id {
        Some(_) => Value::Integer(data.meta.runtime_length.clone()),
        None => {
            let static_len = data
                .meta
                .runtime_length
                .static_val
                .unwrap_or(old_total as i64);
            b.ir_constant_int(static_len)
        }
    };
    let mut prod = b.ir_constant_int(1);
    for v in &shape_values {
        prod = b.ir_mul_i(&prod, v);
    }
    let eq = b.ir_equal_i(&prod, &length_v);
    b.ir_assert(&eq);

    let new_strides = row_major_strides(&logical_shape);
    // Envelope: static axes are exact; each runtime axis is bounded by
    // old_total / product(static dims) (every other runtime dim is >= 1).
    // Consumers (e.g. axis reductions) size their enumeration grids from
    // these bounds — total_bound alone is hopelessly loose for rank > 2.
    let static_product: usize = dim_args
        .iter()
        .filter_map(|v| v.int_val())
        .filter(|&c| c > 0)
        .map(|c| c as usize)
        .product::<usize>()
        .max(1);
    let runtime_axis_bound = (old_total / static_product).max(1);
    require_runtime_dims_within(b, dim_args, &shape_values, &logical_shape, runtime_axis_bound, 0);
    let new_dims: Vec<crate::types::Dim> = dim_args
        .iter()
        .zip(logical_shape.iter())
        .map(|(arg, &s)| match arg.int_val() {
            Some(c) if c > 0 => crate::types::Dim::new_static(&mut b.dim_table, c as usize),
            _ => crate::types::Dim::new_dynamic(&mut b.dim_table, 0, runtime_axis_bound.max(s)),
        })
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, old_total);

    // True runtime strides (C-order over the RUNTIME dims, as IR cells):
    // the logical skeleton's strides describe the padding rectangle, not
    // the actual layout — consumers gathering with runtime geometry (e.g.
    // axis reductions over a runtime-reshaped array) need these. A
    // padded-prefix source instead keeps its envelope prefix strides
    // (see padded_prefix_view).
    let padded = padded_prefix_view(b, data, &shape_values);
    if padded.is_none() && super::padded_with_holes(data) {
        super::refuse_padded_with_holes("reshape");
    }
    let (rt_strides, runtime_length) = match &padded {
        Some((strides, length, _)) => (strides.clone(), length.clone()),
        None => {
            let mut rt_strides: Vec<ScalarValue<i64>> =
                vec![ScalarValue::new(Some(1), None); n_dims];
            let mut stride_acc = b.ir_constant_int(1);
            for i in (0..n_dims).rev() {
                rt_strides[i] = value_to_scalar_i64(&stride_acc);
                if i > 0 {
                    stride_acc = b.ir_mul_i(&stride_acc, &shape_values[i]);
                }
            }
            // The element count is the source's runtime length (asserted
            // equal to the product of the new runtime dims above), not the
            // envelope size: `x[x > 0].reshape(-1)` keeps only the kept
            // elements active.
            (rt_strides, value_to_scalar_i64(&length_v))
        }
    };

    let out = Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id, // compact OR padded view; strides decide
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(n_dims as i64), None),
            runtime_shape,
            runtime_strides: rt_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    });
    if let (Some((_, _, bv)), Some(vid)) = (padded, out.value_id()) {
        b.provenance.set_block_view(vid, bv);
    }
    out
}

/// Reshape a dynamic array to a new static shape.
///
/// Since transpose now materializes, every array is always contiguous
/// row-major. Reshape is therefore a pure metadata operation — the
/// underlying segment is reused as-is.
///
/// Requires: all shape elements are compile-time constants, and
/// `product(new_shape) == product(old_shape)`. One dimension may be -1
/// to infer from the remainder.
pub fn dyn_reshape(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    // Collect target shape as either constant ints or runtime Values.
    // `Some(c)` = compile-time constant, `None(v)` = runtime value (carry the Value through).
    let dim_args: Vec<&Value> = if args.len() == 1 {
        match &args[0] {
            Value::Tuple(d) | Value::List(d) => d.values.iter().collect(),
            v => vec![v],
        }
    } else {
        args.iter().collect()
    };

    let any_runtime = dim_args.iter().any(|v| v.int_val().is_none());
    let has_neg_one = dim_args.iter().any(|v| v.int_val() == Some(-1));
    let old_total: usize = data.meta.logical_shape.iter().product();
    // A source whose runtime_length is witnessed (not just a static constant)
    // is a variable-length dyn array (e.g. boolean-mask gather, sliced dyn).
    // `reshape(-1, ...)` on such a source must infer the unknown dim from
    // `runtime_length`, NOT from the envelope `old_total` — otherwise the
    // emitted "prod(shape) == old_total" assertion fails whenever the runtime
    // length is shorter than the envelope bound.
    let source_runtime_len_is_witness = data.meta.runtime_length.stmt_id.is_some();

    // The reshape paths below reuse the source's segment as a pure
    // metadata change ("always contiguous" precondition). If the source
    // is a stride-permuted view (e.g. from a lazy `dyn_transpose`),
    // materialise into a fresh contiguous segment first.
    let data_owned = super::force_contiguous(b, data);
    let data = &data_owned;

    // Runtime-dim path: source must have a statically-known total (i.e. it's a
    // promoted static array, or a dyn array with bounded total). Lower to a
    // DynamicNDArray with `total_bound = old_total` and runtime shape carried
    // as ScalarValues. Emit an assert that prod(runtime_shape) == total_bound.
    if any_runtime {
        if has_neg_one {
            return dyn_reshape_runtime_infer_unknown(b, data, &dim_args, old_total);
        }
        return dyn_reshape_runtime(b, data, &dim_args, old_total);
    }
    // All dims are compile-time constants, but if any is `-1` and the source
    // is a variable-length dyn array, fall through to the runtime-aware
    // infer_unknown path so the assertion uses `runtime_length` instead of the
    // static envelope bound.
    if has_neg_one && source_runtime_len_is_witness {
        return dyn_reshape_runtime_infer_unknown(b, data, &dim_args, old_total);
    }

    // The static-shape path reuses the buffer as a C-order sequence.
    if super::padded_with_holes(data) {
        super::refuse_padded_with_holes("reshape");
    }
    let raw_shape: Vec<i64> = dim_args
        .iter()
        .map(|v| require_provable_static_int(b, v, SiteKind::ReshapeDim))
        .collect();

    let new_shape = infer_static_reshape_shape(&raw_shape, old_total);

    let new_total: usize = new_shape.iter().product();
    assert_eq!(
        old_total, new_total,
        "reshape: cannot reshape array of size {} into shape {:?} (size {})",
        old_total, new_shape, new_total
    );

    // Length-conformance assert. When the source `runtime_length` is a
    // compile-time constant the check is fully static: fold it to a
    // compile-time comparison (emit nothing on success, a compile error on
    // mismatch) instead of an unconditional in-circuit `assert(== )`.
    if let Some(v) = data.meta.runtime_length.static_val {
        assert_eq!(
            v as usize, new_total,
            "reshape: cannot reshape array of runtime length {} into shape {:?} (size {})",
            v, new_shape, new_total
        );
        // Both sides are compile-time constant and equal — no runtime
        // assertion needed.
    } else {
        let runtime_length_v = if let Some(ptr) = data.meta.runtime_length.stmt_id {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(old_total as i64)
        };
        let new_total_v = b.ir_constant_int(new_total as i64);
        let length_matches = b.ir_equal_i(&runtime_length_v, &new_total_v);
        b.ir_assert(&length_matches);
    }

    let new_strides = row_major_strides(&new_shape);
    let new_dims: Vec<crate::types::Dim> = new_shape
        .iter()
        .map(|&s| crate::types::Dim::new_static(&mut b.dim_table, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, data.envelope.total_bound);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id, // reuse segment — always contiguous
        meta: DynArrayMeta {
            logical_shape: new_shape.clone(),
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: ScalarValue::new(Some(new_shape.len() as i64), None),
            runtime_shape: new_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: new_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

fn infer_static_reshape_shape(raw_shape: &[i64], total: usize) -> Vec<usize> {
    let neg_count = raw_shape.iter().filter(|&&d| d == -1).count();
    assert!(
        neg_count <= 1,
        "reshape: can only specify one unknown dimension (-1)"
    );

    if neg_count == 1 {
        let known_product: usize = raw_shape
            .iter()
            .filter(|&&d| d != -1)
            .map(|&d| {
                assert!(d > 0, "reshape: dimensions must be positive (or -1)");
                d as usize
            })
            .product();
        assert!(
            known_product > 0 && total % known_product == 0,
            "reshape: cannot infer -1 dimension: {} elements, known product {}",
            total,
            known_product
        );
        let inferred = total / known_product;
        raw_shape
            .iter()
            .map(|&d| if d == -1 { inferred } else { d as usize })
            .collect()
    } else {
        raw_shape
            .iter()
            .map(|&d| {
                assert!(d > 0, "reshape: dimensions must be positive (or -1)");
                d as usize
            })
            .collect()
    }
}

/// Require every runtime dimension of a reshape to lie in
/// `[min, bound]`, the extent its envelope axis is built with (`bound` =
/// `max(runtime_axis_bound, logical extent)`). The product assertion alone
/// accepts `(-1) * (-4) == 4` and `0 * 1000 == 0`: NumPy rejects negative
/// dimensions, and an extent past the envelope makes the `0 <= extent <=
/// max` facts recorded for `.shape` false. Constant dimensions are checked
/// at compile time by the callers (an inferred `-1` is pinned by the
/// product); `dims` are the reshape arguments and `shape_values` the
/// matching IR values.
fn require_runtime_dims_within(
    b: &mut IRBuilder,
    dims: &[&Value],
    shape_values: &[Value],
    logical_shape: &[usize],
    runtime_axis_bound: usize,
    min: i64,
) {
    for (axis, dim) in dims.iter().enumerate() {
        if dim.int_val().is_some() {
            continue;
        }
        let bound = runtime_axis_bound.max(logical_shape[axis]) as i64;
        crate::optim::resolver::discharge_index_in_range(
            b,
            &shape_values[axis],
            min,
            bound + 1,
            "reshape_runtime_dimension",
        );
    }
}

fn dyn_reshape_runtime_infer_unknown(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    dim_args: &[&Value],
    old_total: usize,
) -> Value {
    let neg_count = dim_args.iter().filter(|v| v.int_val() == Some(-1)).count();
    assert!(
        neg_count <= 1,
        "reshape: can only specify one unknown dimension (-1)"
    );

    let mut runtime_shape: Vec<ScalarValue<i64>> = Vec::with_capacity(dim_args.len());
    let mut shape_values: Vec<Value> = Vec::with_capacity(dim_args.len());
    let mut logical_shape: Vec<usize> = Vec::with_capacity(dim_args.len());
    let mut unknown_axis: Option<usize> = None;
    let mut const_product: usize = 1;
    let mut runtime_divisor: Option<Value> = None;

    for (axis, arg) in dim_args.iter().enumerate() {
        match arg.int_val() {
            Some(-1) => {
                assert!(
                    unknown_axis.is_none(),
                    "reshape: can only specify one unknown dimension (-1)"
                );
                unknown_axis = Some(axis);
                logical_shape.push(1);
            }
            Some(c) => {
                assert!(c > 0, "reshape: dimensions must be positive (or -1)");
                let c_val = b.ir_constant_int(c);
                runtime_shape.push(ScalarValue::new(Some(c), None));
                shape_values.push(c_val.clone());
                logical_shape.push(c as usize);
                const_product = const_product
                    .checked_mul(c as usize)
                    .expect("reshape: const dim product overflow");
                runtime_divisor = Some(match runtime_divisor {
                    Some(prev) => b.ir_mul_i(&prev, &c_val),
                    None => c_val,
                });
            }
            None => {
                let scalar = match arg {
                    Value::Integer(s) => s.clone(),
                    other => match b.ir_int_cast(other) {
                        Value::Integer(s) => s,
                        _ => unreachable!("ir_int_cast returns Value::Integer"),
                    },
                };
                let scalar_term = Value::Integer(scalar.clone());
                runtime_shape.push(scalar.clone());
                shape_values.push(scalar_term.clone());
                logical_shape.push(1);
                runtime_divisor = Some(match runtime_divisor {
                    Some(prev) => b.ir_mul_i(&prev, &scalar_term),
                    None => scalar_term,
                });
            }
        }
    }

    let unknown_axis = unknown_axis.expect("reshape: unknown dimension not found");
    let runtime_divisor = runtime_divisor.unwrap_or_else(|| b.ir_constant_int(1));
    // Resolve against the array's runtime length (slice-aware), not the
    // parent's static envelope bound. Falls back to old_total when the
    // length wasn't tracked (full, never-sliced arrays).
    let length_v = match data.meta.runtime_length.stmt_id {
        Some(_) => Value::Integer(data.meta.runtime_length.clone()),
        None => {
            let static_len = data
                .meta
                .runtime_length
                .static_val
                .unwrap_or(old_total as i64);
            b.ir_constant_int(static_len)
        }
    };
    // The `-1` dim is `inferred = length / divisor`, pinned downstream by the
    // `prod(all_dims) == length` reconstruction assert below. Every quantity
    // here is bounded by the source total: a valid divisor divides it, the
    // length never exceeds the storage envelope, and so the quotient is at most
    // the total. Emit the bound-certified `FloorDivIPinned` so the Noir backend
    // range-checks via cheap bit-decomposition (~3*bits constraints) instead of
    // the general `q_floor_div`'s two ~126-bit `to_signed` projections. `bits`
    // is `bit_length(old_total) + 1` — a structural, witness-independent
    // constant comfortably above the largest legal quotient/divisor. Other
    // backends treat the op exactly like `FloorDivI`.
    let divisor_bits = (u64::BITS - (old_total as u64).leading_zeros()).max(1) + 1;
    let inferred_v = b.ir_floor_div_i_pinned(&length_v, &runtime_divisor, divisor_bits);
    let inferred_sv = match &inferred_v {
        Value::Integer(s) => s.clone(),
        _ => unreachable!("ir_floor_div_i_pinned returns Value::Integer"),
    };
    runtime_shape.insert(unknown_axis, inferred_sv.clone());
    shape_values.insert(unknown_axis, inferred_v.clone());

    assert!(
        const_product > 0 && old_total % const_product == 0,
        "reshape: const dims ({:?}) don't divide source total {}",
        const_product,
        old_total,
    );

    // Layout invariant (provenance plan #1/#3): prod(logical_shape) must
    // equal the storage total — the skeleton describes the padding
    // rectangle, and chained reshapes / consumers read
    // `prod(logical_shape)` as the buffer size. Pool the runtime
    // variability into the FIRST non-static axis, mirroring
    // `dyn_reshape_runtime`. (Without this, `reshape(h//r, -1, r, c)`
    // left prod(logical) == r*c and a later `.reshape(h, w)` saw a
    // 12-cell source for a 72-cell buffer — case 0389.)
    if let Some(ax) = dim_args
        .iter()
        .position(|v| !matches!(v.int_val(), Some(c) if c > 0))
    {
        logical_shape[ax] = old_total / const_product;
    }

    // Single divisibility assertion: prod(all_dims) == runtime_length.
    // Previously this fired twice — once via `runtime_divisor * inferred ==
    // length` and once via the full-shape product. Both check the same
    // equation (inferred = length / runtime_divisor by construction), so
    // emit only the full-shape form. Cuts ~50% of the reshape's IR overhead
    // and the matching gates downstream.
    let mut prod = b.ir_constant_int(1);
    for v in &shape_values {
        prod = b.ir_mul_i(&prod, v);
    }
    let eq = b.ir_equal_i(&prod, &length_v);
    b.ir_assert(&eq);

    let new_strides = row_major_strides(&logical_shape);
    // Envelope: static axes are exact; each runtime axis is bounded by
    // old_total / product(static dims) (every other runtime dim is >= 1).
    // Consumers (e.g. axis reductions) size their enumeration grids from
    // these bounds — total_bound alone is hopelessly loose for rank > 2.
    let static_product: usize = dim_args
        .iter()
        .filter_map(|v| v.int_val())
        .filter(|&c| c > 0)
        .map(|c| c as usize)
        .product::<usize>()
        .max(1);
    let runtime_axis_bound = (old_total / static_product).max(1);
    // NumPy rejects a zero dimension next to `-1`; with the given ones at
    // least 1 the inferred one is at most the bound.
    require_runtime_dims_within(b, dim_args, &shape_values, &logical_shape, runtime_axis_bound, 1);
    let new_dims: Vec<crate::types::Dim> = dim_args
        .iter()
        .zip(logical_shape.iter())
        .map(|(arg, &s)| match arg.int_val() {
            Some(c) if c > 0 => crate::types::Dim::new_static(&mut b.dim_table, c as usize),
            _ => crate::types::Dim::new_dynamic(&mut b.dim_table, 0, runtime_axis_bound.max(s)),
        })
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, old_total);

    // True runtime strides (C-order over the RUNTIME dims, as IR cells) —
    // see dyn_reshape_runtime: the logical skeleton's strides describe the
    // padding rectangle, not the actual layout. A padded-prefix source
    // keeps its envelope prefix strides (padded_prefix_view).
    let n_dims_total = shape_values.len();
    let padded = padded_prefix_view(b, data, &shape_values);
    if padded.is_none() && super::padded_with_holes(data) {
        super::refuse_padded_with_holes("reshape");
    }
    let (rt_strides, runtime_length) = match &padded {
        Some((strides, length, _)) => (strides.clone(), length.clone()),
        None => {
            let mut rt_strides: Vec<ScalarValue<i64>> =
                vec![ScalarValue::new(Some(1), None); n_dims_total];
            let mut stride_acc = b.ir_constant_int(1);
            for i in (0..n_dims_total).rev() {
                rt_strides[i] = value_to_scalar_i64(&stride_acc);
                if i > 0 {
                    stride_acc = b.ir_mul_i(&stride_acc, &shape_values[i]);
                }
            }
            // The element count is the source's runtime length (asserted
            // equal to the product of the new runtime dims above), not the
            // envelope size: `x[x > 0].reshape(-1)` keeps only the kept
            // elements active.
            (rt_strides, value_to_scalar_i64(&length_v))
        }
    };

    let out = Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id,
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(dim_args.len() as i64), None),
            runtime_shape,
            runtime_strides: rt_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    });
    if let (Some((_, _, bv)), Some(vid)) = (padded, out.value_id()) {
        b.provenance.set_block_view(vid, bv);
    }
    out
}
