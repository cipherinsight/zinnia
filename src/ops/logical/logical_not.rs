use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;
use super::ensure_bool;

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
