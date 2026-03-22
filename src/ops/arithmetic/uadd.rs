use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct UAddOp;

impl UAddOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for UAddOp {
    fn name(&self) -> &'static str { "uadd" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        args.require("x").clone()
    }
}
