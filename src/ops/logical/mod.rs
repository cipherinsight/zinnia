//! Logical operators: and, or, not, xor.
//! Ports `zinnia/op_def/arithmetic/op_logical_*.py`.

use crate::builder::IRBuilder;
use crate::types::Value;

/// Cast a value to boolean if it isn't already.
pub(crate) fn ensure_bool(builder: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::Boolean(_) => val.clone(),
        Value::Integer(_) => builder.ir_bool_cast(val),
        _ => panic!("Cannot cast {:?} to boolean", val.zinnia_type()),
    }
}

pub mod logical_and;
pub mod logical_or;
pub mod logical_not;
pub mod logical_xor;

pub use logical_and::LogicalAndOp;
pub use logical_or::LogicalOrOp;
pub use logical_not::LogicalNotOp;
pub use logical_xor::LogicalXorOp;
