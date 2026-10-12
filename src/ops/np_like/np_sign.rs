use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpSignOp;

impl Op for NpSignOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_sign_i(x),
            Value::Float(_) => builder.ir_sign_f(x),
            _ => panic!("np.sign: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
