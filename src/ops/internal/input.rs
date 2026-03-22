use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::{Value, ZinniaType};

pub struct InputOp {
    pub indices: Vec<u32>,
    pub dtype: ZinniaType,
    pub kind: String, // "PUBLIC", "PRIVATE", etc.
}

impl InputOp {
    const PARAMS: [ParamEntry; 0] = [];
}

impl Op for InputOp {
    fn name(&self) -> &'static str { "input" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        let is_public = self.kind == "PUBLIC";
        match &self.dtype {
            ZinniaType::Integer => {
                builder.ir_read_integer(self.indices.clone(), is_public)
            }
            ZinniaType::Float => {
                builder.ir_read_float(self.indices.clone(), is_public)
            }
            ZinniaType::PoseidonHashed { .. } => {
                builder.ir_read_hash(self.indices.clone(), is_public)
            }
            _ => panic!("input: unsupported type {:?}", self.dtype),
        }
    }
}
