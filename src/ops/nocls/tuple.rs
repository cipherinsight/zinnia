use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct TupleOp;

impl TupleOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for TupleOp {
    fn name(&self) -> &'static str { "tuple" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Tuple(_) => x.clone(),
            Value::List(data) => Value::Tuple(crate::types::CompositeData {
                elements_type: data.elements_type.clone(),
                values: data.values.clone(),
            }),
            _ => panic!("tuple: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
