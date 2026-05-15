//! Boolean-mask classification for dynamic ndarray subscript.
//!
//! The actual masked-read primitive lives in [`super::memory::filter`];
//! this module owns the classification predicate that decides whether a
//! 1-element subscript index is a boolean-mask (delegate to `filter`)
//! versus a fancy-index integer array (delegate to the fancy-index path).

use crate::types::Value;

/// Check if a value looks like a boolean mask.
/// Only `Value::Boolean` leaves qualify — `Value::Integer(0/1)` is fancy indexing.
pub fn is_boolean_mask(val: &Value) -> bool {
    match val {
        Value::DynamicNDArray(_) => true,
        Value::List(d) | Value::Tuple(d) => {
            d.values.iter().all(|v| match v {
                Value::Boolean(_) => true,
                Value::List(_) | Value::Tuple(_) => is_boolean_mask(v),
                _ => false,
            })
        }
        _ => false,
    }
}
