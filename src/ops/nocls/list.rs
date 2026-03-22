use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct ListOp;

impl ListOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for ListOp {
    fn name(&self) -> &'static str { "list" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(_) => x.clone(),
            Value::Tuple(data) => Value::List(crate::types::CompositeData {
                elements_type: data.elements_type.clone(),
                values: data.values.clone(),
            }),
            _ => panic!("list: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
