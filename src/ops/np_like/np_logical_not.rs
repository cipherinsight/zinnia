use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct NpLogicalNotOp;

impl NpLogicalNotOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for NpLogicalNotOp {
    fn name(&self) -> &'static str { "logical_not" }
    fn signature(&self) -> &'static str { "np.logical_not" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Boolean(_) => builder.ir_logical_not(x),
            Value::Integer(_) => {
                let b = builder.ir_bool_cast(x);
                builder.ir_logical_not(&b)
            }
            _ => panic!("np.logical_not: unsupported type"),
        }
    }
}
