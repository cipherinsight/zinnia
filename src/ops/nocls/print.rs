use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct PrintOp;

impl PrintOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for PrintOp {
    fn name(&self) -> &'static str { "print" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        // Print requires a condition and string value
        // The condition comes from OpArgs
        let cond = args.condition.as_ref().cloned()
            .unwrap_or_else(|| builder.ir_constant_bool(true));
        let str_val = match x {
            Value::String(_) => x.clone(),
            Value::Integer(_) | Value::Boolean(_) => builder.ir_str_i(x),
            Value::Float(_) => builder.ir_str_f(x),
            _ => panic!("print: unsupported type {:?}", x.zinnia_type()),
        };
        builder.ir_print(&cond, &str_val)
    }
}
