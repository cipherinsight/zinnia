use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;
use super::ensure_bool;

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
