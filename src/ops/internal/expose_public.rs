use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct ExposePublicOp;

impl ExposePublicOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for ExposePublicOp {
    fn name(&self) -> &'static str { "expose_public" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_expose_public_i(x),
            Value::Float(_) => builder.ir_expose_public_f(x),
            _ => panic!("expose_public: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
