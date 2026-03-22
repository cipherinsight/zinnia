use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct IntCastOp;

impl IntCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for IntCastOp {
    fn name(&self) -> &'static str { "int" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) => x.clone(),
            Value::Boolean(_) => x.clone(), // Bool is already integer-like
            Value::Float(_) => builder.ir_int_cast(x),
            _ => panic!("int(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
