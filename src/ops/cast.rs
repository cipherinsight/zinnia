//! Cast operators: int, float, bool, str.
//! Ports `zinnia/op_def/nocls/op_int_cast.py`, `op_float_cast.py`, `op_bool_cast.py`.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct IntCastOp;

impl IntCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for IntCastOp {
    fn name(&self) -> &'static str { "int" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) => x.clone(),
            Value::Boolean(_) => x.clone(), // Bool is already integer-like
            Value::Float(_) => builder.ir_int_cast(x),
            _ => panic!("int(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct FloatCastOp;

impl FloatCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for FloatCastOp {
    fn name(&self) -> &'static str { "float" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Float(_) => x.clone(),
            Value::Integer(_) | Value::Boolean(_) => builder.ir_float_cast(x),
            _ => panic!("float(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct BoolCastOp;

impl BoolCastOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for BoolCastOp {
    fn name(&self) -> &'static str { "bool" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Boolean(_) => x.clone(),
            Value::Integer(_) => builder.ir_bool_cast(x),
            _ => panic!("bool(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct StrOp;

impl StrOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for StrOp {
    fn name(&self) -> &'static str { "str" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_str_i(x),
            Value::Float(_) => builder.ir_str_f(x),
            Value::String(_) => x.clone(),
            _ => panic!("str(): unsupported type {:?}", x.zinnia_type()),
        }
    }
}
