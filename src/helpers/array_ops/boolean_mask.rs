//! Boolean-mask classification for dynamic ndarray subscript.
//!
//! The actual masked-read primitive lives in [`super::memory::filter`];
//! this module owns the classification predicate that decides whether a
//! 1-element subscript index is a boolean-mask (delegate to `filter`)
//! versus a fancy-index integer array (delegate to the fancy-index path).

use crate::types::Value;

/// Check if a value looks like a boolean mask.
/// Only `Value::Boolean` leaves qualify — `Value::Integer(0/1)` is fancy indexing.
/// Builder-aware variant: also recognizes segment-backed StaticArray
/// masks (P4a list literals / np.asarray of bools) via the payload-cell
/// probe.
pub fn is_boolean_mask_with(b: &mut crate::builder::IRBuilder, val: &Value) -> bool {
    if matches!(val, Value::StaticArray { .. }) {
        return crate::helpers::static_array::mask::is_boolean_mask_static_array(b, val);
    }
    is_boolean_mask(val)
}

pub fn is_boolean_mask(val: &Value) -> bool {
    match val {
        Value::DynamicNDArray(_) => true,
        Value::List(d) | Value::Tuple(d) => d.values.iter().all(|v| match v {
            Value::Boolean(_) => true,
            Value::List(_) | Value::Tuple(_) => is_boolean_mask(v),
            _ => false,
        }),
        _ => false,
    }
}
