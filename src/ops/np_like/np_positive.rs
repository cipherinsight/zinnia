use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpPositiveOp;

impl Op for NpPositiveOp {
    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        args.require("x").clone()
    }
}
