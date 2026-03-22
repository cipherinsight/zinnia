use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct PoseidonHashBuiltinOp;

impl PoseidonHashBuiltinOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for PoseidonHashBuiltinOp {
    fn name(&self) -> &'static str { "poseidon_hash" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        builder.ir_poseidon_hash(&[x.clone()])
    }
}
