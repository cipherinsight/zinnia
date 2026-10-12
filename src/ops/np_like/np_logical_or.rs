use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpLogicalOrOp;

impl Op for NpLogicalOrOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        builder.ir_logical_or(x1, x2)
    }
}
