use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct NpLogicalOrOp;

impl NpLogicalOrOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("x1"),
        ParamEntry::required("x2"),
    ];
}

impl Op for NpLogicalOrOp {
    fn name(&self) -> &'static str { "logical_or" }
    fn signature(&self) -> &'static str { "np.logical_or" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        builder.ir_logical_or(x1, x2)
    }
}
