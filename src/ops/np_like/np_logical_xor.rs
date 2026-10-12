use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpLogicalXorOp;

impl Op for NpLogicalXorOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        let or_val = builder.ir_logical_or(x1, x2);
        let and_val = builder.ir_logical_and(x1, x2);
        let not_and = builder.ir_logical_not(&and_val);
        builder.ir_logical_and(&or_val, &not_and)
    }
}
