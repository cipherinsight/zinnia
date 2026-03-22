use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct NpSignOp;

impl NpSignOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for NpSignOp {
    fn name(&self) -> &'static str { "sign" }
    fn signature(&self) -> &'static str { "np.sign" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_sign_i(x),
            Value::Float(_) => builder.ir_sign_f(x),
            _ => panic!("np.sign: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
