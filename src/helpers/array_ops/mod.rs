//! Unified array operations that handle both static and dynamic ndarrays.
//!
//! Each function is a single entry point for one operation. It inspects the
//! value type (static composite vs DynamicNDArray) and parameter dynamism,
//! then routes to the appropriate implementation. Static operators can call
//! dynamic counterparts directly — no centralized dispatcher needed.

mod array_index;
mod assignment;
mod bounded_axis;
mod indexing;
mod memory;
mod reduce;
mod shape;

pub use assignment::{dyn_setitem, dyn_setitem_mask, dyn_setitem_slice, setitem_runtime_positions};
pub use array_index::{
    array_index_kind, is_bool_array, leading_index_array, reject_unsupported_index_arrays,
    ArrayIndexKind,
};
pub(crate) use bounded_axis::{select_stride_mode, StrideMode};
pub use indexing::dyn_subscript;
pub(crate) use indexing::python_slice_bounds;
pub use memory::{filter, repeat};
pub use reduce::{argmax_argmin, reduce};
pub use shape::{moveaxis, reshape, swapaxes, transpose};

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
