use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;
use super::ensure_bool;

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
