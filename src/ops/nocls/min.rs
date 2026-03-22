use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct MinOp;

impl MinOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("a"),
        ParamEntry::required("b"),
    ];
}

impl Op for MinOp {
    fn name(&self) -> &'static str { "min" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let a = args.require("a");
        let b = args.require("b");
        // min(a, b) = select(a < b, a, b)
        match (a, b) {
            (Value::Integer(_), Value::Integer(_))
            | (Value::Boolean(_), Value::Integer(_))
            | (Value::Integer(_), Value::Boolean(_))
            | (Value::Boolean(_), Value::Boolean(_)) => {
                let cond = builder.ir_less_than_i(a, b);
                builder.ir_select_i(&cond, a, b)
            }
            (Value::Float(_), Value::Float(_)) => {
                let cond = builder.ir_less_than_f(a, b);
                builder.ir_select_f(&cond, a, b)
            }
            _ => panic!("min: unsupported types {:?} and {:?}", a.zinnia_type(), b.zinnia_type()),
        }
    }
}
