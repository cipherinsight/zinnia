use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct BoolCastOp;

impl BoolCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for BoolCastOp {
    fn name(&self) -> &'static str { "bool" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Boolean(_) => x.clone(),
            Value::Integer(_) => builder.ir_bool_cast(x),
            _ => panic!("bool(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
