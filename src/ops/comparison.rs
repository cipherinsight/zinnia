//! Comparison operators: eq, ne, lt, lte, gt, gte.
//! Ports `zinnia/op_def/arithmetic/op_eq.py`, `op_ne.py`, `op_lt.py`, etc.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

/// Helper: dispatch a comparison op. Always returns Boolean.
pub(crate) fn compare_number_op_pub(
    builder: &mut IRBuilder,
    lhs: &Value,
    rhs: &Value,
    int_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
    float_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
) -> Value {
    compare_number_op(builder, lhs, rhs, int_op, float_op)
}

fn compare_number_op(
    builder: &mut IRBuilder,
    lhs: &Value,
    rhs: &Value,
    int_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
    float_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
) -> Value {
    match (lhs, rhs) {
        (Value::Integer(_) | Value::Boolean(_), Value::Integer(_) | Value::Boolean(_)) => {
            int_op(builder, lhs, rhs)
        }
        (Value::Float(_), Value::Float(_)) => float_op(builder, lhs, rhs),
        (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
            let lf = builder.ir_float_cast(lhs);
            float_op(builder, &lf, rhs)
        }
        (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
            let rf = builder.ir_float_cast(rhs);
            float_op(builder, lhs, &rf)
        }
        _ => panic!(
            "compare_number_op: unsupported types {:?} and {:?}",
            lhs.zinnia_type(),
            rhs.zinnia_type()
        ),
    }
}

macro_rules! define_compare_op {
    ($name:ident, $op_name:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;

        impl $name {
            const PARAMS: [ParamEntry; 2] = [
                ParamEntry::required("lhs"),
                ParamEntry::required("rhs"),
            ];
        }

        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let lhs = args.require("lhs");
                let rhs = args.require("rhs");

                if lhs.is_number() && rhs.is_number() {
                    return compare_number_op(
                        builder,
                        lhs,
                        rhs,
                        IRBuilder::$int_method,
                        IRBuilder::$float_method,
                    );
                }

                // TODO: NDArray comparison
                panic!("Op `{}` not supported for {:?} and {:?}",
                    $op_name, lhs.zinnia_type(), rhs.zinnia_type())
            }
        }
    };
}

define_compare_op!(EqualOp, "eq", ir_equal_i, ir_equal_f);
define_compare_op!(NotEqualOp, "ne", ir_not_equal_i, ir_not_equal_f);
define_compare_op!(LessThanOp, "lt", ir_less_than_i, ir_less_than_f);
define_compare_op!(LessThanOrEqualOp, "lte", ir_less_than_or_equal_i, ir_less_than_or_equal_f);
define_compare_op!(GreaterThanOp, "gt", ir_greater_than_i, ir_greater_than_f);
define_compare_op!(GreaterThanOrEqualOp, "gte", ir_greater_than_or_equal_i, ir_greater_than_or_equal_f);
