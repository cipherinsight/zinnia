//! Unified array operations that handle both static and dynamic ndarrays.
//!
//! Each function is a single entry point for one operation. It inspects the
//! value type (static composite vs DynamicNDArray) and parameter dynamism,
//! then routes to the appropriate implementation. Static operators can call
//! dynamic counterparts directly — no centralized dispatcher needed.

mod reduce;
mod shape;
mod memory;
mod indexing;
mod assignment;

pub use reduce::{reduce, argmax_argmin};
pub use shape::{transpose, moveaxis, reshape, swapaxes};
pub use memory::{filter, repeat};
pub use indexing::{dyn_subscript, is_boolean_mask};
pub use assignment::{dyn_setitem, dyn_setitem_mask, dyn_setitem_slice};

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::DynAggKind;
use crate::types::Value;

pub(crate) fn promote(b: &mut IRBuilder, val: &Value) -> crate::types::DynamicNDArrayData {
    let promoted = crate::helpers::promote::promote_static_to_dynamic(b, val);
    match promoted {
        Value::DynamicNDArray(d) => d,
        _ => unreachable!(),
    }
}

pub(crate) fn agg_kind(op: &str) -> DynAggKind {
    match op {
        "sum" => DynAggKind::Sum,
        "prod" => DynAggKind::Prod,
        "max" => DynAggKind::Max,
        "min" => DynAggKind::Min,
        "all" => DynAggKind::All,
        "any" => DynAggKind::Any,
        "argmax" => DynAggKind::Argmax,
        "argmin" => DynAggKind::Argmin,
        _ => panic!("unknown aggregation op: {}", op),
    }
}
