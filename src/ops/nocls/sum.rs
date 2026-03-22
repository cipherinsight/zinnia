use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct SumOp;

impl SumOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for SumOp {
    fn name(&self) -> &'static str { "sum" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            // Scalar passthrough: sum of a single number is itself.
            Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => x.clone(),
            Value::List(data) => {
                if data.values.is_empty() {
                    return builder.ir_constant_int(0);
                }
                let mut acc = data.values[0].clone();
                for v in &data.values[1..] {
                    acc = crate::ops::dispatch_binary_numeric(
                        builder, &acc, v,
                        IRBuilder::ir_add_i, IRBuilder::ir_add_f,
                    );
                }
                acc
            }
            Value::Tuple(data) => {
                if data.values.is_empty() {
                    return builder.ir_constant_int(0);
                }
                let mut acc = data.values[0].clone();
                for v in &data.values[1..] {
                    acc = crate::ops::dispatch_binary_numeric(
                        builder, &acc, v,
                        IRBuilder::ir_add_i, IRBuilder::ir_add_f,
                    );
                }
                acc
            }
            _ => panic!("sum: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
