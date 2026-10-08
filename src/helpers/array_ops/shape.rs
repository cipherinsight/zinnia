//! Unified shape operations (transpose, moveaxis, reshape, swapaxes).

use crate::builder::IRBuilder;
use crate::optim::resolver::{require_provable_static_int, SiteKind};
use crate::types::{Value, ValueId};

use super::promote;

/// Helper: materialise StaticArray into a Value::List before legacy paths.
fn maybe_materialise_static_array(b: &mut IRBuilder, val: &Value) -> Value {
    if matches!(val, Value::StaticArray { .. }) {
        crate::helpers::static_array::base::to_value_list(b, val)
    } else {
        val.clone()
    }
}

/// Unified transpose.
pub fn transpose(b: &mut IRBuilder, val: &Value, axes_args: &[Value]) -> Value {
    let out = if let Value::DynamicNDArray(d) = val {
        crate::ops::dyn_ndarray::reshape::dyn_transpose(b, d, axes_args)
    } else {
        let has_dynamic = axes_args.iter().any(|a| match a {
            Value::List(d) | Value::Tuple(d) => d.values.iter().any(|v| v.int_val().is_none()),
            v => !matches!(v, Value::None) && v.int_val().is_none(),
        });

        if has_dynamic {
            let val_m = maybe_materialise_static_array(b, val);
            let d = promote(b, &val_m);
            crate::ops::dyn_ndarray::reshape::dyn_transpose(b, &d, axes_args)
        } else if let Some(out) =
            crate::helpers::static_array::shape::try_apply_transpose(b, val, axes_args)
        {
            // P4c: native StaticArray dispatch before legacy boundary.
            out
        } else {
            let val_m = maybe_materialise_static_array(b, val);
            crate::helpers::ndarray::ndarray_transpose(b, &val_m, axes_args)
        }
    };
    out
}

/// Unified moveaxis.
pub fn moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let out = if let Value::DynamicNDArray(d) = val {
        crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, d, args)
    } else {
        let has_dynamic = args.iter().any(|v| v.int_val().is_none());
        if has_dynamic {
            let val_m = maybe_materialise_static_array(b, val);
            let d = promote(b, &val_m);
            crate::ops::dyn_ndarray::reshape::dyn_moveaxis(b, &d, args)
        } else if let Some(out) =
            crate::helpers::static_array::shape::try_apply_moveaxis(b, val, args)
        {
            // P4c: native StaticArray dispatch before legacy boundary.
            out
        } else {
            let val_m = maybe_materialise_static_array(b, val);
            crate::ops::static_ndarray_ops::ndarray_moveaxis(b, &val_m, args)
        }
    };
    out
}

/// Unified reshape.
pub fn reshape(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let out = if let Value::DynamicNDArray(d) = val {
        crate::ops::dyn_ndarray::reshape::dyn_reshape(b, d, args)
    } else {
        let has_dynamic = args.iter().any(|a| match a {
            Value::List(d) | Value::Tuple(d) => d.values.iter().any(|v| v.int_val().is_none()),
            v => v.int_val().is_none(),
        });

        if has_dynamic {
            // Keep a dynamic-bounds slice's segment identity (and its
            // runtime-extents record) through the promotion: listifying
            // re-claims the full static envelope, and the runtime
            // reshape then either mis-addresses the padded buffer or
            // rejects valid witnesses on its length assert.
            let d = if matches!(val, Value::StaticArray { segment_id, .. }
                if b.provenance.runtime_extents_of(*segment_id).is_some())
            {
                match crate::helpers::promote::promote_static_to_dynamic_with_extents(b, val) {
                    Value::DynamicNDArray(d) => d,
                    _ => unreachable!("with_extents promote returns a dyn array"),
                }
            } else {
                let val_m = maybe_materialise_static_array(b, val);
                promote(b, &val_m)
            };
            crate::ops::dyn_ndarray::reshape::dyn_reshape(b, &d, args)
        } else if let Some(out) =
            crate::helpers::static_array::shape::try_apply_reshape(b, val, args)
        {
            // P4c: native StaticArray dispatch before legacy boundary.
            out
        } else {
            let val_m = maybe_materialise_static_array(b, val);
            crate::ops::static_ndarray_ops::ndarray_reshape(b, &val_m, args)
        }
    };
    out
}

/// Unified swapaxes.
pub fn swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let out = if let Value::DynamicNDArray(d) = val {
        let ndim = d.envelope.rank();
        let a0_i: i64 = require_provable_static_int(b, &args[0], SiteKind::Axis);
        let a1_i: i64 = require_provable_static_int(b, &args[1], SiteKind::Axis);
        let a0 = a0_i as usize;
        let a1 = a1_i as usize;
        let perm: Vec<Value> = (0..ndim)
            .map(|i| {
                let j = if i == a0 {
                    a1
                } else if i == a1 {
                    a0
                } else {
                    i
                };
                Value::Integer(crate::types::ScalarValue::new(Some(j as i64), None))
            })
            .collect();
        let perm_list = Value::List(crate::types::CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; ndim],
            values: perm,

            value_id: ValueId::next(),
        });
        crate::ops::dyn_ndarray::reshape::dyn_transpose(b, d, &[perm_list])
    } else {
        let has_dynamic = args.iter().any(|v| v.int_val().is_none());
        if has_dynamic {
            let d = promote(b, val);
            // Recursive call handles its own relay; return its result directly.
            let out = swapaxes(b, &Value::DynamicNDArray(d), args);
            return out;
        }

        crate::ops::static_ndarray_ops::ndarray_swapaxes(b, val, args)
    };
    out
}
