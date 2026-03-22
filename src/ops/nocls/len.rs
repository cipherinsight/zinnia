use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct LenOp;

impl LenOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for LenOp {
    fn name(&self) -> &'static str { "len" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => builder.ir_constant_int(data.values.len() as i64),
            Value::Tuple(data) => builder.ir_constant_int(data.values.len() as i64),
            Value::NDArray(data) => builder.ir_constant_int(data.shape[0] as i64),
            _ => panic!("len: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
