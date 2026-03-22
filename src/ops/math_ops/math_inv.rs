use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct MathInvOp;

impl MathInvOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for MathInvOp {
    fn name(&self) -> &'static str { "inv" }
    fn signature(&self) -> &'static str { "math.inv" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_inv_i(x),
            _ => panic!("math.inv: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
