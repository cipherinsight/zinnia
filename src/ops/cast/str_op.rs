use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct StrOp;

impl StrOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for StrOp {
    fn name(&self) -> &'static str { "str" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_str_i(x),
            Value::Float(_) => builder.ir_str_f(x),
            Value::String(_) => x.clone(),
            _ => panic!("str(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
