//! Unified shape operations (transpose, moveaxis, reshape, swapaxes).

use crate::builder::IRBuilder;
use crate::optim::resolver::{require_static_int, SiteKind};
use crate::types::Value;

use super::promote;

/// Helper: materialise StaticArray into a Value::List before legacy paths.
fn maybe_materialise_static_array(b: &mut IRBuilder, val: &Value) -> Value {
    if matches!(val, Value::StaticArray { .. }) {
        crate::helpers::static_array::to_value_list(b, val)
    } else {
        val.clone()
    }
}

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
        let val = maybe_materialise_static_array(b, val);
        let d = promote(b, &val);
        crate::ops::dyn_ndarray::reshape::dyn_transpose(b, &d, axes_args)
    } else {
        // P4c: native StaticArray dispatch before legacy boundary.
        if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(b, val, axes_args) {
            return out;
        }
        let val = maybe_materialise_static_array(b, val);
        crate::helpers::ndarray::ndarray_transpose(b, &val, axes_args)
    }
}

/// Unified moveaxis.
pub fn moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, d, args);
    }

    let has_dynamic = args.iter().any(|v| v.int_val().is_none());
    if has_dynamic {
        let val = maybe_materialise_static_array(b, val);
        let d = promote(b, &val);
        crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, &d, args)
    } else {
        // P4c: native StaticArray dispatch before legacy boundary.
        if let Some(out) = crate::helpers::static_array_shape::try_apply_moveaxis(b, val, args) {
            return out;
        }
        let val = maybe_materialise_static_array(b, val);
        crate::ops::static_ndarray_ops::ndarray_moveaxis(b, &val, args)
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
        let val = maybe_materialise_static_array(b, val);
        let d = promote(b, &val);
        crate::ops::dyn_ndarray::reshape::dyn_reshape(b, &d, args)
    } else {
        // P4c: native StaticArray dispatch before legacy boundary.
        if let Some(out) = crate::helpers::static_array_shape::try_apply_reshape(b, val, args) {
            return out;
        }
        let val = maybe_materialise_static_array(b, val);
        crate::ops::static_ndarray_ops::ndarray_reshape(b, &val, args)
    }
}

/// Unified swapaxes.
pub fn swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    if let Value::DynamicNDArray(d) = val {
        let ndim = d.envelope.rank();
        let a0_i: i64 = require_static_int(b, &args[0], SiteKind::Axis, None)
            .unwrap_or_else(|e| panic!("{}", e.message))
            .into();
        let a1_i: i64 = require_static_int(b, &args[1], SiteKind::Axis, None)
            .unwrap_or_else(|e| panic!("{}", e.message))
            .into();
        let a0 = a0_i as usize;
        let a1 = a1_i as usize;
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
