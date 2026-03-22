// DivOp is NOT defined by macro — Python `/` always returns float.
// Both operands are cast to float before dividing, matching Python semantics.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct DivOp;

impl DivOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("lhs"),
        ParamEntry::required("rhs"),
    ];
}

impl Op for DivOp {
    fn name(&self) -> &'static str { "div" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let lhs = args.require("lhs");
        let rhs = args.require("rhs");

        if lhs.is_number() && rhs.is_number() {
            // Always cast to float — Python `/` never returns int.
            let lf = match lhs {
                Value::Float(_) => lhs.clone(),
                _ => builder.ir_float_cast(lhs),
            };
            let rf = match rhs {
                Value::Float(_) => rhs.clone(),
                _ => builder.ir_float_cast(rhs),
            };
            return builder.ir_div_f(&lf, &rf);
        }

        panic!("Op `div` not supported for {:?} and {:?}",
            lhs.zinnia_type(), rhs.zinnia_type())
    }
}
