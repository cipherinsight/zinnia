use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpLogicalNotOp;

impl Op for NpLogicalNotOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Boolean(_) => builder.ir_logical_not(x),
            Value::Integer(_) => {
                let b = builder.ir_bool_cast(x);
                builder.ir_logical_not(&b)
            }
            _ => panic!("np.logical_not: unsupported type"),
        }
    }
}
