use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::{Value, ZinniaType};

pub struct PlaceholderValueOp {
    pub dtype: ZinniaType,
}

impl PlaceholderValueOp {
    const PARAMS: [ParamEntry; 0] = [];
}

impl Op for PlaceholderValueOp {
    fn name(&self) -> &'static str { "placeholder_value" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        match &self.dtype {
            ZinniaType::Integer => builder.ir_constant_int(0),
            ZinniaType::Float => builder.ir_constant_float(0.0),
            ZinniaType::Boolean => builder.ir_constant_bool(false),
            _ => panic!("placeholder_value: unsupported type {:?}", self.dtype),
        }
    }
}
