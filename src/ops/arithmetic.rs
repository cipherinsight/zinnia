//! Arithmetic operators: add, sub, mul, div, floor_divide, mod, power, usub, uadd, mat_mul.
//! Ports `zinnia/op_def/arithmetic/`.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

/// Helper: dispatch a binary number op based on the types of lhs and rhs.
/// Handles Int+Int, Float+Float, and mixed Int+Float (promotes to Float).
pub(crate) fn binary_number_op_pub(
    builder: &mut IRBuilder,
    lhs: &Value,
    rhs: &Value,
    int_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
    float_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
) -> Value {
    binary_number_op(builder, lhs, rhs, int_op, float_op)
}

fn binary_number_op(
    builder: &mut IRBuilder,
    lhs: &Value,
    rhs: &Value,
    int_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
    float_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
) -> Value {
    match (lhs, rhs) {
        (Value::Integer(_), Value::Integer(_))
        | (Value::Boolean(_), Value::Integer(_))
        | (Value::Integer(_), Value::Boolean(_))
        | (Value::Boolean(_), Value::Boolean(_)) => int_op(builder, lhs, rhs),
        (Value::Float(_), Value::Float(_)) => float_op(builder, lhs, rhs),
        (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
            let lf = builder.ir_float_cast(lhs);
            float_op(builder, &lf, rhs)
        }
        (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
            let rf = builder.ir_float_cast(rhs);
            float_op(builder, lhs, &rf)
        }
        _ => panic!(
            "binary_number_op: unsupported types {:?} and {:?}",
            lhs.zinnia_type(),
            rhs.zinnia_type()
        ),
    }
}

// Macro for defining binary arithmetic ops
macro_rules! define_binary_arith_op {
    ($name:ident, $op_name:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;

        impl $name {
            const PARAMS: [ParamEntry; 2] = [
                ParamEntry::required("lhs"),
                ParamEntry::required("rhs"),
            ];
        }

        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let lhs = args.require("lhs");
                let rhs = args.require("rhs");

                // Scalar path
                if lhs.is_number() && rhs.is_number() {
                    return binary_number_op(
                        builder,
                        lhs,
                        rhs,
                        IRBuilder::$int_method,
                        IRBuilder::$float_method,
                    );
                }

                // TODO: NDArray, List, Tuple, String paths
                panic!("Op `{}` not supported for {:?} and {:?}",
                    $op_name, lhs.zinnia_type(), rhs.zinnia_type())
            }
        }
    };
}

// AddOp is NOT defined by macro — needs string concat and composite support.

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
            return binary_number_op(builder, lhs, rhs, IRBuilder::ir_add_i, IRBuilder::ir_add_f);
        }

        panic!("Op `add` not supported for {:?} and {:?}",
            lhs.zinnia_type(), rhs.zinnia_type())
    }
}
define_binary_arith_op!(SubOp, "sub", ir_sub_i, ir_sub_f);
define_binary_arith_op!(MulOp, "mul", ir_mul_i, ir_mul_f);
// DivOp is NOT defined by macro — Python `/` always returns float.
// Both operands are cast to float before dividing, matching Python semantics.

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

define_binary_arith_op!(FloorDivideOp, "floor_divide", ir_floor_div_i, ir_floor_div_f);
define_binary_arith_op!(ModOp, "mod", ir_mod_i, ir_mod_f);
define_binary_arith_op!(PowerOp, "power", ir_pow_i, ir_pow_f);
define_binary_arith_op!(MatMulOp, "matmul", ir_mul_i, ir_mul_f); // placeholder

// ── Unary operators ───────────────────────────────────────────────────

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

pub struct UAddOp;

impl UAddOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for UAddOp {
    fn name(&self) -> &'static str { "uadd" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        args.require("x").clone()
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
