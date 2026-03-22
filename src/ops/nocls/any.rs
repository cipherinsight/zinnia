use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AnyOp;

impl AnyOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for AnyOp {
    fn name(&self) -> &'static str { "any" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => {
                let mut acc = builder.ir_constant_bool(false);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_or(&acc, &b);
                }
                acc
            }
            Value::Tuple(data) => {
                let mut acc = builder.ir_constant_bool(false);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_or(&acc, &b);
                }
                acc
            }
            _ => panic!("any: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
