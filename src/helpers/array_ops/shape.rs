//! Unified shape operations (transpose, moveaxis, reshape, swapaxes).

use crate::builder::IRBuilder;
use crate::types::Value;

use super::promote;

/// Unified transpose.
pub fn transpose(b: &mut IRBuilder, val: &Value, axes_args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::reshape::dyn_transpose(b, d, axes_args);
    }

    let has_dynamic = axes_args.iter().any(|a| match a {
        Value::List(d) | Value::Tuple(d) => d.values.iter().any(|v| v.int_val().is_none()),
        v => !matches!(v, Value::None) && v.int_val().is_none(),
    });

    if has_dynamic {
        let d = promote(b, val);
        crate::ops::dyn_ndarray::reshape::dyn_transpose(b, &d, axes_args)
    } else {
        crate::helpers::ndarray::ndarray_transpose(b, val, axes_args)
    }
}

/// Unified moveaxis.
pub fn moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, d, args);
    }

    let has_dynamic = args.iter().any(|v| v.int_val().is_none());
    if has_dynamic {
        let d = promote(b, val);
        crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, &d, args)
    } else {
        crate::ops::static_ndarray_ops::ndarray_moveaxis(b, val, args)
    }
}

/// Unified reshape.
pub fn reshape(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::reshape::dyn_reshape(b, d, args);
    }

    let has_dynamic = args.iter().any(|a| match a {
        Value::List(d) | Value::Tuple(d) => d.values.iter().any(|v| v.int_val().is_none()),
        v => v.int_val().is_none(),
    });

    if has_dynamic {
        let d = promote(b, val);
        crate::ops::dyn_ndarray::reshape::dyn_reshape(b, &d, args)
    } else {
        crate::ops::static_ndarray_ops::ndarray_reshape(b, val, args)
    }
}

/// Unified swapaxes.
pub fn swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        let ndim = d.envelope.rank();
        let a0 = args[0].int_val().expect("swapaxes: axis must be constant for dynamic arrays") as usize;
        let a1 = args[1].int_val().expect("swapaxes: axis must be constant for dynamic arrays") as usize;
        let perm: Vec<Value> = (0..ndim).map(|i| {
            let j = if i == a0 { a1 } else if i == a1 { a0 } else { i };
            Value::Integer(crate::types::ScalarValue::new(Some(j as i64), None))
        }).collect();
        let perm_list = Value::List(crate::types::CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; ndim],
            values: perm,
        });
        return crate::ops::dyn_ndarray::reshape::dyn_transpose(b, d, &[perm_list]);
    }

    let has_dynamic = args.iter().any(|v| v.int_val().is_none());
    if has_dynamic {
        let d = promote(b, val);
        return swapaxes(b, &Value::DynamicNDArray(d), args);
    }

    crate::ops::static_ndarray_ops::ndarray_swapaxes(b, val, args)
}
