//! DynamicNDArray method operators.
//! Ports `zinnia/op_def/dynamic_ndarray/`.
//!
//! DynamicNDArray methods mirror NDArray methods but operate on memory-backed
//! arrays whose size is not known at compile time. These are dispatched via
//! ir_gen.rs and the memory subsystem (AllocateMemory, ReadMemory, WriteMemory).

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

macro_rules! define_dyn_ndarray_stub {
    ($name:ident, $op_name:expr, $sig:expr, $params:expr) => {
        pub struct $name;
        impl $name {
            const PARAMS: &'static [ParamEntry] = $params;
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[ParamEntry] { Self::PARAMS }
            fn build(&self, _builder: &mut IRBuilder, _args: &OpArgs) -> Value {
                panic!("{} not yet implemented in Rust backend", $sig);
            }
        }
    };
}

define_dyn_ndarray_stub!(DynNDArraySumOp, "sum", "DynamicNDArray.sum", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayProdOp, "prod", "DynamicNDArray.prod", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayMaxOp, "max", "DynamicNDArray.max", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayMinOp, "min", "DynamicNDArray.min", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayArgmaxOp, "argmax", "DynamicNDArray.argmax", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayArgminOp, "argmin", "DynamicNDArray.argmin", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayAllOp, "all", "DynamicNDArray.all", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayAnyOp, "any", "DynamicNDArray.any", &[ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayFlattenOp, "flatten", "DynamicNDArray.flatten", &[]);
define_dyn_ndarray_stub!(DynNDArrayTOp, "T", "DynamicNDArray.T", &[]);
define_dyn_ndarray_stub!(DynNDArrayTransposeOp, "transpose", "DynamicNDArray.transpose", &[ParamEntry::optional("axes")]);
define_dyn_ndarray_stub!(DynNDArrayMoveaxisOp, "moveaxis", "DynamicNDArray.moveaxis", &[ParamEntry::required("source"), ParamEntry::required("destination")]);
define_dyn_ndarray_stub!(DynNDArrayAstypeOp, "astype", "DynamicNDArray.astype", &[ParamEntry::required("dtype")]);
define_dyn_ndarray_stub!(DynNDArrayDtypeOp, "dtype", "DynamicNDArray.dtype", &[]);
define_dyn_ndarray_stub!(DynNDArrayShapeOp, "shape", "DynamicNDArray.shape", &[]);
define_dyn_ndarray_stub!(DynNDArraySizeOp, "size", "DynamicNDArray.size", &[]);
define_dyn_ndarray_stub!(DynNDArrayNdimOp, "ndim", "DynamicNDArray.ndim", &[]);
define_dyn_ndarray_stub!(DynNDArrayFlatOp, "flat", "DynamicNDArray.flat", &[]);
define_dyn_ndarray_stub!(DynNDArrayFilterOp, "filter", "DynamicNDArray.filter", &[ParamEntry::required("mask")]);
define_dyn_ndarray_stub!(DynNDArrayTolistOp, "tolist", "DynamicNDArray.tolist", &[]);
define_dyn_ndarray_stub!(DynNDArrayGetItemOp, "__get_item__", "DynamicNDArray.__get_item__", &[ParamEntry::required("key")]);
define_dyn_ndarray_stub!(DynNDArraySetItemOp, "__set_item__", "DynamicNDArray.__set_item__", &[ParamEntry::required("key"), ParamEntry::required("value")]);
define_dyn_ndarray_stub!(DynNDArrayRepeatOp, "repeat", "DynamicNDArray.repeat", &[ParamEntry::required("repeats"), ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayConcatenateOp, "concatenate", "DynamicNDArray.concatenate", &[ParamEntry::required("other"), ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayStackOp, "stack", "DynamicNDArray.stack", &[ParamEntry::required("other"), ParamEntry::optional("axis")]);
define_dyn_ndarray_stub!(DynNDArrayZerosOp, "zeros", "DynamicNDArray.zeros", &[ParamEntry::required("shape"), ParamEntry::optional("dtype")]);
define_dyn_ndarray_stub!(DynNDArrayOnesOp, "ones", "DynamicNDArray.ones", &[ParamEntry::required("shape"), ParamEntry::optional("dtype")]);
define_dyn_ndarray_stub!(DynNDArrayEyeOp, "eye", "DynamicNDArray.eye", &[ParamEntry::required("N"), ParamEntry::optional("M")]);
