#![allow(clippy::cloned_ref_to_slice_refs)]

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct SelectOp;

impl SelectOp {
    const PARAMS: [ParamEntry; 3] = [
        ParamEntry::required("cond"),
        ParamEntry::required("tv"),
        ParamEntry::required("fv"),
    ];
}

impl Op for SelectOp {
    fn name(&self) -> &'static str { "select" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let cond = args.require("cond");
        let tv = args.require("tv");
        let fv = args.require("fv");

        match (tv, fv) {
            (Value::Boolean(_), Value::Boolean(_)) => builder.ir_select_b(cond, tv, fv),
            (Value::Integer(_), Value::Integer(_))
            | (Value::Boolean(_), Value::Integer(_))
            | (Value::Integer(_), Value::Boolean(_)) => builder.ir_select_i(cond, tv, fv),
            (Value::Float(_), Value::Float(_)) => builder.ir_select_f(cond, tv, fv),
            (Value::None, Value::None) => Value::None,
            _ => panic!(
                "select: unsupported types {:?} and {:?}",
                tv.zinnia_type(),
                fv.zinnia_type()
            ),
        }
    }
}
