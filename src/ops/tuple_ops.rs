//! Tuple method operators: count, index.
//! Ports `zinnia/op_def/tuple_ops/`.
//!
//! NOTE: Tuple methods are handled directly in `ir_gen.rs` via the visitor pattern
//! (they share the list_method_count / list_method_index implementations).

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

pub struct TupleCountOp;
impl TupleCountOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("value")];
}
impl Op for TupleCountOp {
    fn name(&self) -> &'static str { "count" }
    fn signature(&self) -> &'static str { "Tuple.count" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, _builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        panic!("Tuple.count is dispatched via ir_gen.rs, not the Op registry");
    }
}

pub struct TupleIndexOp;
impl TupleIndexOp {
    const PARAMS: [ParamEntry; 3] = [
        ParamEntry::required("value"),
        ParamEntry::optional("start"),
        ParamEntry::optional("stop"),
    ];
}
impl Op for TupleIndexOp {
    fn name(&self) -> &'static str { "index" }
    fn signature(&self) -> &'static str { "Tuple.index" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, _builder: &mut IRBuilder, _args: &OpArgs) -> Value {
        panic!("Tuple.index is dispatched via ir_gen.rs, not the Op registry");
    }
}
