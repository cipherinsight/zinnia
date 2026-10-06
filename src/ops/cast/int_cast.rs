use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct IntCastOp;

impl IntCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for IntCastOp {
    fn name(&self) -> &'static str {
        "int"
    }
    fn params(&self) -> &[ParamEntry] {
        &Self::PARAMS
    }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            // `int(flag)` of a bool is an integer, not a bool cell (an
            // array of them indexes by position).
            Value::Integer(_) | Value::Boolean(_) => builder.int_dtype_cell(x),
            Value::Float(_) => builder.ir_int_cast(x),
            _ => panic!("int(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
