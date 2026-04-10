use crate::builder::IRBuilder;
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, ScalarValue, Value, ZinniaType,
};

pub fn dyn_transpose(_b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
    let ndim = shape.len();

    if ndim <= 1 {
        return Value::DynamicNDArray(data.clone());
    }

    // Determine axis permutation
    let perm: Vec<usize> =
        if args.is_empty() || matches!(args.first(), Some(Value::None)) {
            // Default: reverse all axes
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
            // Multiple int args as axes
            args.iter()
                .map(|v| {
                    let a = v.int_val().expect("transpose: axes must be constant ints");
                    let resolved = if a < 0 { ndim as i64 + a } else { a };
                    resolved as usize
                })
                .collect()
        };

    assert_eq!(perm.len(), ndim, "transpose: permutation length must match rank");

    // Permute shape and strides
    let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let new_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();
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

    // Permute the envelope's dims by the same axis permutation. Dim vars
    // are preserved (not freshly allocated), so any unifications established
    // before the transpose stay valid after.
    let new_dims: Vec<crate::types::Dim> = perm.iter().map(|&p| data.envelope.dims[p]).collect();
    // Shape-preserving: total_bound is conserved from source (§3.2).
    let envelope = crate::types::Envelope::new_with_bound(new_dims, data.envelope.total_bound);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id: data.segment_id,    // same underlying segment (view op)
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
