//! Logical operators: and, or, not, xor.
//! Ports `zinnia/op_def/arithmetic/op_logical_*.py`.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct LogicalAndOp;

impl LogicalAndOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("lhs"),
        ParamEntry::required("rhs"),
    ];
}

impl Op for LogicalAndOp {
    fn name(&self) -> &'static str { "logical_and" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let lhs = args.require("lhs");
        let rhs = args.require("rhs");
        let lhs_bool = ensure_bool(builder, lhs);
        let rhs_bool = ensure_bool(builder, rhs);
        builder.ir_logical_and(&lhs_bool, &rhs_bool)
    }
}

pub struct LogicalOrOp;

impl LogicalOrOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("lhs"),
        ParamEntry::required("rhs"),
    ];
}

impl Op for LogicalOrOp {
    fn name(&self) -> &'static str { "logical_or" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let lhs = args.require("lhs");
        let rhs = args.require("rhs");
        let lhs_bool = ensure_bool(builder, lhs);
        let rhs_bool = ensure_bool(builder, rhs);
        builder.ir_logical_or(&lhs_bool, &rhs_bool)
    }
}

pub struct LogicalNotOp;

impl LogicalNotOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for LogicalNotOp {
    fn name(&self) -> &'static str { "logical_not" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        let x_bool = ensure_bool(builder, x);
        builder.ir_logical_not(&x_bool)
    }
}

pub struct LogicalXorOp;

impl LogicalXorOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("lhs"),
        ParamEntry::required("rhs"),
    ];
}

impl Op for LogicalXorOp {
    fn name(&self) -> &'static str { "logical_xor" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let lhs = args.require("lhs");
        let rhs = args.require("rhs");
        let lhs_bool = ensure_bool(builder, lhs);
        let rhs_bool = ensure_bool(builder, rhs);
        // XOR = (A OR B) AND NOT (A AND B)
        let or_val = builder.ir_logical_or(&lhs_bool, &rhs_bool);
        let and_val = builder.ir_logical_and(&lhs_bool, &rhs_bool);
        let not_and = builder.ir_logical_not(&and_val);
        builder.ir_logical_and(&or_val, &not_and)
    }
}

/// Cast a value to boolean if it isn't already.
fn ensure_bool(builder: &mut IRBuilder, val: &Value) -> Value {
    match val {
        Value::Boolean(_) => val.clone(),
        Value::Integer(_) => builder.ir_bool_cast(val),
        _ => panic!("Cannot cast {:?} to boolean", val.zinnia_type()),
    }
}
