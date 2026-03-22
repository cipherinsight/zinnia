use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct NpLogicalXorOp;

impl NpLogicalXorOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("x1"),
        ParamEntry::required("x2"),
    ];
}

impl Op for NpLogicalXorOp {
    fn name(&self) -> &'static str { "logical_xor" }
    fn signature(&self) -> &'static str { "np.logical_xor" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        let or_val = builder.ir_logical_or(x1, x2);
        let and_val = builder.ir_logical_and(x1, x2);
        let not_and = builder.ir_logical_not(&and_val);
        builder.ir_logical_and(&or_val, &not_and)
    }
}
