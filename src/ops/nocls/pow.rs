use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct PowOp;

impl PowOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("base"),
        ParamEntry::required("exp"),
    ];
}

impl Op for PowOp {
    fn name(&self) -> &'static str { "pow" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let base = args.require("base");
        let exp = args.require("exp");
        match (base, exp) {
            (Value::Integer(_), Value::Integer(_)) => builder.ir_pow_i(base, exp),
            (Value::Float(_), Value::Float(_)) => builder.ir_pow_f(base, exp),
            (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
                let bf = builder.ir_float_cast(base);
                builder.ir_pow_f(&bf, exp)
            }
            (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
                let ef = builder.ir_float_cast(exp);
                builder.ir_pow_f(base, &ef)
            }
            _ => panic!("pow: unsupported types"),
        }
    }
}
