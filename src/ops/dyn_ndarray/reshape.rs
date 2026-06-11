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

    let new_runtime_strides: Vec<ScalarValue<i64>> = perm
        .iter()
        .map(|&p| {
            if p < data.meta.runtime_strides.len() {
                data.meta.runtime_strides[p].clone()
            } else {
                ScalarValue::new(Some(strides[p] as i64), None)
            }
        })
        .collect();

    Value::DynamicNDArray(DynamicNDArrayData {
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
    let new_dims: Vec<crate::types::Dim> = logical_shape
        .iter()
        .map(|&s| crate::types::Dim::new_static(&mut b.dim_table, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, old_total);

    // True runtime strides (C-order over the RUNTIME dims, as IR cells):
    // the logical skeleton's strides describe the padding rectangle, not
    // the actual layout — consumers gathering with runtime geometry (e.g.
    // axis reductions over a runtime-reshaped array) need these.
    let mut rt_strides: Vec<ScalarValue<i64>> = vec![ScalarValue::new(Some(1), None); n_dims];
    let mut stride_acc = b.ir_constant_int(1);
    for i in (0..n_dims).rev() {
        rt_strides[i] = value_to_scalar_i64(&stride_acc);
        if i > 0 {
            stride_acc = b.ir_mul_i(&stride_acc, &shape_values[i]);
        }
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id, // segment is contiguous; reuse
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: ScalarValue::new(Some(old_total as i64), None),
            runtime_rank: ScalarValue::new(Some(n_dims as i64), None),
            runtime_shape,
            runtime_strides: rt_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
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

    let runtime_length_v = if let Some(v) = data.meta.runtime_length.static_val {
        b.ir_constant_int(v)
    } else if let Some(ptr) = data.meta.runtime_length.stmt_id {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(old_total as i64)
    };
    let new_total_v = b.ir_constant_int(new_total as i64);
    let length_matches = b.ir_equal_i(&runtime_length_v, &new_total_v);
    b.ir_assert(&length_matches);

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
    let inferred_v = b.ir_floor_div_i(&length_v, &runtime_divisor);
    let inferred_sv = match &inferred_v {
        Value::Integer(s) => s.clone(),
        _ => unreachable!("ir_floor_div_i returns Value::Integer"),
    };
    runtime_shape.insert(unknown_axis, inferred_sv.clone());
    shape_values.insert(unknown_axis, inferred_v.clone());

    assert!(
        const_product > 0 && old_total % const_product == 0,
        "reshape: const dims ({:?}) don't divide source total {}",
        const_product,
        old_total,
    );

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
    let new_dims: Vec<crate::types::Dim> = logical_shape
        .iter()
        .map(|&s| crate::types::Dim::new_static(&mut b.dim_table, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, old_total);

    // True runtime strides (C-order over the RUNTIME dims, as IR cells) —
    // see dyn_reshape_runtime: the logical skeleton's strides describe the
    // padding rectangle, not the actual layout.
    let n_dims_total = shape_values.len();
    let mut rt_strides: Vec<ScalarValue<i64>> =
        vec![ScalarValue::new(Some(1), None); n_dims_total];
    let mut stride_acc = b.ir_constant_int(1);
    for i in (0..n_dims_total).rev() {
        rt_strides[i] = value_to_scalar_i64(&stride_acc);
        if i > 0 {
            stride_acc = b.ir_mul_i(&stride_acc, &shape_values[i]);
        }
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id,
        meta: DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: ScalarValue::new(Some(old_total as i64), None),
            runtime_rank: ScalarValue::new(Some(dim_args.len() as i64), None),
            runtime_shape,
            runtime_strides: rt_strides,
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}
