use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs};
use crate::types::Value;

pub struct NpNegativeOp;

impl Op for NpNegativeOp {
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => {
                let zero = builder.ir_constant_int(0);
                builder.ir_sub_i(&zero, x)
            }
            Value::Float(_) => {
                let zero = builder.ir_constant_float(0.0);
                builder.ir_sub_f(&zero, x)
            }
            _ => panic!("np.negative: unsupported type"),
        }
    }
}
