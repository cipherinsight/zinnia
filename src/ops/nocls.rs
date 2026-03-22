//! No-class (global) operators: abs, len, min, max, sum, any, all, range, print,
//! pow, list, tuple, poseidon_hash_builtin, merkle_verify.
//! Ports `zinnia/op_def/nocls/`.

#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct AbsOp;

impl AbsOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for AbsOp {
    fn name(&self) -> &'static str { "abs" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_abs_i(x),
            Value::Float(_) => builder.ir_abs_f(x),
            _ => panic!("abs: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct LenOp;

impl LenOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for LenOp {
    fn name(&self) -> &'static str { "len" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => builder.ir_constant_int(data.values.len() as i64),
            Value::Tuple(data) => builder.ir_constant_int(data.values.len() as i64),
            Value::NDArray(data) => builder.ir_constant_int(data.shape[0] as i64),
            _ => panic!("len: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct MinOp;

impl MinOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("a"),
        ParamEntry::required("b"),
    ];
}

impl Op for MinOp {
    fn name(&self) -> &'static str { "min" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let a = args.require("a");
        let b = args.require("b");
        // min(a, b) = select(a < b, a, b)
        match (a, b) {
            (Value::Integer(_), Value::Integer(_))
            | (Value::Boolean(_), Value::Integer(_))
            | (Value::Integer(_), Value::Boolean(_))
            | (Value::Boolean(_), Value::Boolean(_)) => {
                let cond = builder.ir_less_than_i(a, b);
                builder.ir_select_i(&cond, a, b)
            }
            (Value::Float(_), Value::Float(_)) => {
                let cond = builder.ir_less_than_f(a, b);
                builder.ir_select_f(&cond, a, b)
            }
            _ => panic!("min: unsupported types {:?} and {:?}", a.zinnia_type(), b.zinnia_type()),
        }
    }
}

pub struct MaxOp;

impl MaxOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("a"),
        ParamEntry::required("b"),
    ];
}

impl Op for MaxOp {
    fn name(&self) -> &'static str { "max" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let a = args.require("a");
        let b = args.require("b");
        match (a, b) {
            (Value::Integer(_), Value::Integer(_))
            | (Value::Boolean(_), Value::Integer(_))
            | (Value::Integer(_), Value::Boolean(_))
            | (Value::Boolean(_), Value::Boolean(_)) => {
                let cond = builder.ir_greater_than_i(a, b);
                builder.ir_select_i(&cond, a, b)
            }
            (Value::Float(_), Value::Float(_)) => {
                let cond = builder.ir_greater_than_f(a, b);
                builder.ir_select_f(&cond, a, b)
            }
            _ => panic!("max: unsupported types {:?} and {:?}", a.zinnia_type(), b.zinnia_type()),
        }
    }
}

pub struct PrintOp;

impl PrintOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for PrintOp {
    fn name(&self) -> &'static str { "print" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        // Print requires a condition and string value
        // The condition comes from OpArgs
        let cond = args.condition.as_ref().cloned()
            .unwrap_or_else(|| builder.ir_constant_bool(true));
        let str_val = match x {
            Value::String(_) => x.clone(),
            Value::Integer(_) | Value::Boolean(_) => builder.ir_str_i(x),
            Value::Float(_) => builder.ir_str_f(x),
            _ => panic!("print: unsupported type {:?}", x.zinnia_type()),
        };
        builder.ir_print(&cond, &str_val)
    }
}

// ListOp, TupleOp, RangeOp, SumOp, AnyOp, AllOp are more complex
// and depend on composite value manipulation. Stubbed for now.

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

pub struct TupleOp;
impl TupleOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for TupleOp {
    fn name(&self) -> &'static str { "tuple" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Tuple(_) => x.clone(),
            Value::List(data) => Value::Tuple(crate::types::CompositeData {
                elements_type: data.elements_type.clone(),
                values: data.values.clone(),
            }),
            _ => panic!("tuple: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct PowOp;
impl PowOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("base"),
        ParamEntry::required("exp"),
    ];
}
impl Op for PowOp {
    fn name(&self) -> &'static str { "pow" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let base = args.require("base");
        let exp = args.require("exp");
        match (base, exp) {
            (Value::Integer(_), Value::Integer(_)) => builder.ir_pow_i(base, exp),
            (Value::Float(_), Value::Float(_)) => builder.ir_pow_f(base, exp),
            (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
                let bf = builder.ir_float_cast(base);
                builder.ir_pow_f(&bf, exp)
            }
            (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
                let ef = builder.ir_float_cast(exp);
                builder.ir_pow_f(base, &ef)
            }
            _ => panic!("pow: unsupported types"),
        }
    }
}

pub struct SumOp;
impl SumOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for SumOp {
    fn name(&self) -> &'static str { "sum" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            // Scalar passthrough: sum of a single number is itself.
            Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => x.clone(),
            Value::List(data) => {
                if data.values.is_empty() {
                    return builder.ir_constant_int(0);
                }
                let mut acc = data.values[0].clone();
                for v in &data.values[1..] {
                    acc = crate::ops::arithmetic::binary_number_op_pub(
                        builder, &acc, v,
                        IRBuilder::ir_add_i, IRBuilder::ir_add_f,
                    );
                }
                acc
            }
            Value::Tuple(data) => {
                if data.values.is_empty() {
                    return builder.ir_constant_int(0);
                }
                let mut acc = data.values[0].clone();
                for v in &data.values[1..] {
                    acc = crate::ops::arithmetic::binary_number_op_pub(
                        builder, &acc, v,
                        IRBuilder::ir_add_i, IRBuilder::ir_add_f,
                    );
                }
                acc
            }
            _ => panic!("sum: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct AnyOp;
impl AnyOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for AnyOp {
    fn name(&self) -> &'static str { "any" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => {
                let mut acc = builder.ir_constant_bool(false);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_or(&acc, &b);
                }
                acc
            }
            Value::Tuple(data) => {
                let mut acc = builder.ir_constant_bool(false);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_or(&acc, &b);
                }
                acc
            }
            _ => panic!("any: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct AllOp;
impl AllOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for AllOp {
    fn name(&self) -> &'static str { "all" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::List(data) => {
                let mut acc = builder.ir_constant_bool(true);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            Value::Tuple(data) => {
                let mut acc = builder.ir_constant_bool(true);
                for v in &data.values {
                    let b = builder.ir_bool_cast(v);
                    acc = builder.ir_logical_and(&acc, &b);
                }
                acc
            }
            _ => panic!("all: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

pub struct RangeOp;
impl RangeOp {
    const PARAMS: [ParamEntry; 3] = [
        ParamEntry::required("start"),
        ParamEntry::optional("stop"),
        ParamEntry::optional("step"),
    ];
}
impl Op for RangeOp {
    fn name(&self) -> &'static str { "range" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let start_val = args.require("start");
        let stop_opt = args.get("stop");
        let step_opt = args.get("step");

        // All range arguments must be statically known integers.
        let (start, stop, step) = if let Some(stop_v) = stop_opt {
            let s = start_val.int_val().expect("range: start must be a constant integer");
            let e = stop_v.int_val().expect("range: stop must be a constant integer");
            let st = step_opt
                .and_then(|v| v.int_val())
                .unwrap_or(1);
            (s, e, st)
        } else {
            // range(stop) — start=0, stop=start_val
            let e = start_val.int_val().expect("range: stop must be a constant integer");
            (0, e, 1)
        };

        assert!(step != 0, "range: step must not be zero");

        let mut values = Vec::new();
        let mut types = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            values.push(builder.ir_constant_int(i));
            types.push(crate::types::ZinniaType::Integer);
            i += step;
        }

        Value::List(crate::types::CompositeData {
            elements_type: types,
            values,
        })
    }
}

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

pub struct MerkleVerifyOp;
impl MerkleVerifyOp {
    const PARAMS: [ParamEntry; 4] = [
        ParamEntry::required("leaf"),
        ParamEntry::required("root"),
        ParamEntry::required("siblings"),
        ParamEntry::required("directions"),
    ];
}
impl Op for MerkleVerifyOp {
    fn name(&self) -> &'static str { "merkle_verify" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let leaf = args.require("leaf");
        let root = args.require("root");
        let siblings = args.require("siblings");
        let directions = args.require("directions");

        let (sib_vals, dir_vals) = match (siblings, directions) {
            (Value::List(s), Value::List(d)) => (&s.values, &d.values),
            (Value::Tuple(s), Value::Tuple(d)) => (&s.values, &d.values),
            (Value::Tuple(s), Value::List(d)) => (&s.values, &d.values),
            (Value::List(s), Value::Tuple(d)) => (&s.values, &d.values),
            _ => panic!("merkle_verify: siblings and directions must be List or Tuple"),
        };
        assert_eq!(
            sib_vals.len(),
            dir_vals.len(),
            "merkle_verify: siblings and directions must have the same length"
        );

        let mut acc = leaf.clone();
        for (sib, dir) in sib_vals.iter().zip(dir_vals.iter()) {
            // dir == 0 means acc is on the left, sib on the right
            // dir == 1 means sib is on the left, acc on the right
            let left = builder.ir_select_i(dir, sib, &acc);
            let right = builder.ir_select_i(dir, &acc, sib);
            acc = builder.ir_poseidon_hash(&[left, right]);
        }

        builder.ir_equal_i(&acc, root)
    }
}
