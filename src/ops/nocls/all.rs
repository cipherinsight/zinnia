use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AllOp;

impl AllOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for AllOp {
    fn name(&self) -> &'static str { "all" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => {
                let mut acc = builder.ir_constant_bool(true);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            Value::Tuple(data) => {
                let mut acc = builder.ir_constant_bool(true);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            _ => panic!("all: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
