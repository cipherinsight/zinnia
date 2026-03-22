use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct USubOp;

impl USubOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for USubOp {
    fn name(&self) -> &'static str { "usub" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => {
                let zero = builder.ir_constant_int(0);
                builder.ir_sub_i(&zero, x)
            }
            Value::Float(_) => {
                let zero = builder.ir_constant_float(0.0);
                builder.ir_sub_f(&zero, x)
            }
            _ => panic!("usub: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_usub_op() {
        let mut b = IRBuilder::new();
        let x = b.ir_constant_int(5);
        let mut kw = HashMap::new();
        kw.insert("x".to_string(), x);
        let args = OpArgs::new(kw);
        let result = USubOp.build(&mut b, &args);
        assert_eq!(result.int_val(), Some(-5));
    }
}
