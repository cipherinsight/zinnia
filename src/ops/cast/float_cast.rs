use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct FloatCastOp;

impl FloatCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for FloatCastOp {
    fn name(&self) -> &'static str { "float" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Float(_) => x.clone(),
            Value::Integer(_) | Value::Boolean(_) => builder.ir_float_cast(x),
            _ => panic!("float(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
