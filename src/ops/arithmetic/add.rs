use crate::builder::IRBuilder;
use crate::ops::{dispatch_binary_numeric, Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AddOp;

impl AddOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("lhs"),
        ParamEntry::required("rhs"),
    ];
}

impl Op for AddOp {
    fn name(&self) -> &'static str { "add" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let lhs = args.require("lhs");
        let rhs = args.require("rhs");

        // String concatenation
        if matches!(lhs, Value::String(_)) && matches!(rhs, Value::String(_)) {
            return builder.ir_add_str(lhs, rhs);
        }

        // List/Tuple concatenation
        match (lhs, rhs) {
            (Value::List(ld), Value::List(rd)) => {
                let mut types = ld.elements_type.clone();
                types.extend(rd.elements_type.clone());
                let mut vals = ld.values.clone();
                vals.extend(rd.values.clone());
                return Value::List(crate::types::CompositeData { elements_type: types, values: vals });
            }
            (Value::Tuple(ld), Value::Tuple(rd)) => {
                let mut types = ld.elements_type.clone();
                types.extend(rd.elements_type.clone());
                let mut vals = ld.values.clone();
                vals.extend(rd.values.clone());
                return Value::Tuple(crate::types::CompositeData { elements_type: types, values: vals });
            }
            _ => {}
        }

        // Scalar numeric path
        if lhs.is_number() && rhs.is_number() {
            return dispatch_binary_numeric(builder, lhs, rhs, IRBuilder::ir_add_i, IRBuilder::ir_add_f);
        }

        panic!("Op `add` not supported for {:?} and {:?}",
            lhs.zinnia_type(), rhs.zinnia_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_add_op_int() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(10);
        let c = b.ir_constant_int(20);
        let mut kw = HashMap::new();
        kw.insert("lhs".to_string(), a);
        kw.insert("rhs".to_string(), c);
        let args = OpArgs::new(kw);
        let result = AddOp.build(&mut b, &args);
        assert_eq!(result.int_val(), Some(30));
    }
}
