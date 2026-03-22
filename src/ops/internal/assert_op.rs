use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AssertOp;

impl AssertOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("test"),
        ParamEntry::optional("condition"),
    ];
}

impl Op for AssertOp {
    fn name(&self) -> &'static str { "assert" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let test = args.require("test");
        let asserted = match test {
            Value::Boolean(_) | Value::Integer(_) => test.clone(),
            _ => panic!("assert: test must be boolean or integer"),
        };
        builder.ir_assert(&asserted)
    }
}
