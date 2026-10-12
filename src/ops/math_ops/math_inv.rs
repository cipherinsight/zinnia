use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct MathInvOp;

impl Op for MathInvOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_inv_i(x),
            _ => panic!("math.inv: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
