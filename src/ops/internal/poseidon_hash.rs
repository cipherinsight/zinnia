use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct PoseidonHashOp;

impl PoseidonHashOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for PoseidonHashOp {
    fn name(&self) -> &'static str { "poseidon_hash" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) | Value::Float(_) => {
                builder.ir_poseidon_hash(&[x.clone()])
            }
            _ => panic!("poseidon_hash: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
