use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct NpPositiveOp;

impl NpPositiveOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for NpPositiveOp {
    fn name(&self) -> &'static str { "positive" }
    fn signature(&self) -> &'static str { "np.positive" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        args.require("x").clone()
    }
}
