//! List method operators: append, extend, insert, pop, remove, clear, index, count, reverse, copy.
//! Ports `zinnia/op_def/list/`.
//!
//! NOTE: List methods are handled directly in `ir_gen.rs` via the visitor pattern.
//! These Op trait implementations exist for registry completeness.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

macro_rules! define_list_stub {
    ($name:ident, $op_name:expr, $sig:expr, $params:expr, $inplace:expr) => {
        pub struct $name;
        impl $name {
            const PARAMS: &'static [ParamEntry] = $params;
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn is_inplace(&self) -> bool { $inplace }
            fn params(&self) -> &[ParamEntry] { Self::PARAMS }
            fn build(&self, _builder: &mut IRBuilder, _args: &OpArgs) -> Value {
                panic!("List.{} is dispatched via ir_gen.rs, not the Op registry", $op_name);
            }
        }
    };
}

define_list_stub!(ListAppendOp, "append", "List.append", &[ParamEntry::required("object")], true);
define_list_stub!(ListExtendOp, "extend", "List.extend", &[ParamEntry::required("iterable")], true);
define_list_stub!(ListInsertOp, "insert", "List.insert", &[ParamEntry::required("index"), ParamEntry::required("object")], true);
define_list_stub!(ListPopOp, "pop", "List.pop", &[ParamEntry::optional("index")], true);
define_list_stub!(ListRemoveOp, "remove", "List.remove", &[ParamEntry::required("value")], true);
define_list_stub!(ListClearOp, "clear", "List.clear", &[], true);
define_list_stub!(ListIndexOp, "index", "List.index", &[ParamEntry::required("value"), ParamEntry::optional("start"), ParamEntry::optional("stop")], false);
define_list_stub!(ListCountOp, "count", "List.count", &[ParamEntry::required("value")], false);
define_list_stub!(ListReverseOp, "reverse", "List.reverse", &[], true);
define_list_stub!(ListCopyOp, "copy", "List.copy", &[], false);
