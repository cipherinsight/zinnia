use crate::builder::IRBuilder;
use crate::helpers::shape_arith::row_major_strides;
use crate::optim::resolver::{require_static_int, SiteKind};
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, ScalarValue, Value, ZinniaType,
};

use super::value_to_scalar_i64;

/// Transpose a dynamic array, materializing into a fresh contiguous segment.
///
/// Previous implementation was a view op (permuted strides, reused segment),
/// but downstream consumers (binary ops, aggregation, etc.) assume contiguous
/// row-major layout when they `read_all` + index by `row_major_strides`.
/// Materializing here (O(N)) eliminates that entire class of bugs and costs
/// nothing extra at proof time (the prover touches every element regardless).
pub fn dyn_transpose(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
    let ndim = shape.len();

    if ndim <= 1 {
        return Value::DynamicNDArray(data.clone());
    }

    // Determine axis permutation
    let perm: Vec<usize> =
        if args.is_empty() || matches!(args.first(), Some(Value::None)) {
            (0..ndim).rev().collect()
        } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) =
            args.first()
        {
            perm_data
                .values
                .iter()
                .map(|v| {
                    let a = v.int_val().expect("transpose: axes must be constant ints");
                    let resolved = if a < 0 { ndim as i64 + a } else { a };
                    resolved as usize
                })
                .collect()
        } else {
            args.iter()
                .map(|v| {
                    let a = v.int_val().expect("transpose: axes must be constant ints");
                    let resolved = if a < 0 { ndim as i64 + a } else { a };
                    resolved as usize
                })
                .collect()
        };

    assert_eq!(perm.len(), ndim, "transpose: permutation length must match rank");

    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides = row_major_strides(&new_shape);
    let out_total: usize = new_shape.iter().product();

    // Materialize: read elements in transposed logical order, write contiguously.
    let src_vals = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let out_strides_for_decode = row_major_strides(&new_shape);

    let mut out_elements = Vec::with_capacity(out_total);
    for flat_out in 0..out_total {
        // Decode output flat index → output coords (in transposed shape)
        let out_coords = crate::helpers::shape_arith::decode_coords(
            flat_out, &new_shape, &out_strides_for_decode,
        );
        // Map back to source coords: out_coords[i] is for axis perm[i]
        // So source coord for original axis perm[i] = out_coords[i]
        let src_flat: usize = (0..ndim)
            .map(|i| out_coords[i] * strides[perm[i]])
            .sum();
        out_elements.push(value_to_scalar_i64(&src_vals[src_flat]));
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);

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

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: new_shape,
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: data.meta.runtime_rank.clone(),
            runtime_shape: new_runtime_shape,
            runtime_strides: new_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    })
}

pub fn dyn_moveaxis(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let ndim = data.meta.logical_shape.len();
    assert!(args.len() >= 2, "moveaxis: requires source and destination");

    let src = {
        let s = args[0]
            .int_val()
            .expect("moveaxis: source must be constant int");
        if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
    };
    let dst = {
        let d = args[1]
            .int_val()
            .expect("moveaxis: destination must be constant int");
        if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
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
    });
    dyn_transpose(b, data, &[axes_tuple])
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
                const_product = const_product.checked_mul(c as usize)
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
            const_product, old_total,
        );
        // Place all the runtime variability into the first runtime axis so
        // prod(logical_shape) == old_total.
        logical_shape[runtime_axes[0]] = old_total / const_product;
    }

    // Assert prod(runtime_shape) == old_total at proof time.
    let mut prod = b.ir_constant_int(1);
    for v in &shape_values {
        prod = b.ir_mul_i(&prod, v);
    }
    let bound = b.ir_constant_int(old_total as i64);
    let eq = b.ir_equal_i(&prod, &bound);
    b.ir_assert(&eq);

    let new_strides = row_major_strides(&logical_shape);
    let new_dims: Vec<crate::types::Dim> = logical_shape
        .iter()
        .map(|&s| crate::types::Dim::new_static(&mut b.dim_table, s))
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(new_dims, old_total);

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
            runtime_strides: new_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
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
    let old_total: usize = data.meta.logical_shape.iter().product();

    // Runtime-dim path: source must have a statically-known total (i.e. it's a
    // promoted static array, or a dyn array with bounded total). Lower to a
    // DynamicNDArray with `total_bound = old_total` and runtime shape carried
    // as ScalarValues. Emit an assert that prod(runtime_shape) == total_bound.
    if any_runtime {
        return dyn_reshape_runtime(b, data, &dim_args, old_total);
    }

    let raw_shape: Vec<i64> = dim_args
        .iter()
        .map(|v| {
            require_static_int(b, v, SiteKind::ReshapeDim, None)
                .unwrap_or_else(|e| panic!("{}", e.message))
                .into()
        })
        .collect();

    // Handle -1 inference.
    let neg_count = raw_shape.iter().filter(|&&d| d == -1).count();
    assert!(neg_count <= 1, "reshape: can only specify one unknown dimension (-1)");

    let new_shape: Vec<usize> = if neg_count == 1 {
        let known_product: usize = raw_shape
            .iter()
            .filter(|&&d| d != -1)
            .map(|&d| {
                assert!(d > 0, "reshape: dimensions must be positive (or -1)");
                d as usize
            })
            .product();
        assert!(
            known_product > 0 && old_total % known_product == 0,
            "reshape: cannot infer -1 dimension: {} elements, known product {}",
            old_total,
            known_product
        );
        let inferred = old_total / known_product;
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
    };

    let new_total: usize = new_shape.iter().product();
    assert_eq!(
        old_total, new_total,
        "reshape: cannot reshape array of size {} into shape {:?} (size {})",
        old_total, new_shape, new_total
    );

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
    })
}
