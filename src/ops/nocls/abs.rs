use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AbsOp;

impl AbsOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for AbsOp {
    fn name(&self) -> &'static str { "abs" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_abs_i(x),
            Value::Float(_) => builder.ir_abs_f(x),
            _ => panic!("abs: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
