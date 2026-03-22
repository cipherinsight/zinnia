//! NDArray method operators: sum, prod, max, min, argmax, argmin, all, any,
//! reshape, flatten, T, transpose, moveaxis, astype, dtype, shape, size, ndim,
//! flat, filter, tolist, get_item, set_item, repeat.
//! Ports `zinnia/op_def/ndarray/`.
//!
//! NOTE: Most NDArray operations are handled directly in `ir_gen.rs` via the
//! visitor pattern. These Op trait implementations exist so the registry can
//! dispatch NDArray-namespaced calls when needed (e.g., from the Python bridge).
//! They are thin stubs that panic — the real work happens in ir_gen.rs.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

macro_rules! define_ndarray_stub {
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
                panic!("NDArray.{} is dispatched via ir_gen.rs, not the Op registry", $op_name);
            }
        }
    };
}

define_ndarray_stub!(NDArraySumOp, "sum", "NDArray.sum", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayProdOp, "prod", "NDArray.prod", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayMaxOp, "max", "NDArray.max", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayMinOp, "min", "NDArray.min", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayArgmaxOp, "argmax", "NDArray.argmax", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayArgminOp, "argmin", "NDArray.argmin", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayAllOp, "all", "NDArray.all", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayAnyOp, "any", "NDArray.any", &[ParamEntry::optional("axis")]);
define_ndarray_stub!(NDArrayReshapeOp, "reshape", "NDArray.reshape", &[ParamEntry::required("shape")]);
define_ndarray_stub!(NDArrayFlattenOp, "flatten", "NDArray.flatten", &[]);
define_ndarray_stub!(NDArrayTOp, "T", "NDArray.T", &[]);
define_ndarray_stub!(NDArrayTransposeOp, "transpose", "NDArray.transpose", &[ParamEntry::optional("axes")]);
define_ndarray_stub!(NDArrayMoveaxisOp, "moveaxis", "NDArray.moveaxis", &[ParamEntry::required("source"), ParamEntry::required("destination")]);
define_ndarray_stub!(NDArrayAstypeOp, "astype", "NDArray.astype", &[ParamEntry::required("dtype")]);
define_ndarray_stub!(NDArrayDtypeOp, "dtype", "NDArray.dtype", &[]);
define_ndarray_stub!(NDArrayShapeOp, "shape", "NDArray.shape", &[]);
define_ndarray_stub!(NDArraySizeOp, "size", "NDArray.size", &[]);
define_ndarray_stub!(NDArrayNdimOp, "ndim", "NDArray.ndim", &[]);
define_ndarray_stub!(NDArrayFlatOp, "flat", "NDArray.flat", &[]);
define_ndarray_stub!(NDArrayFilterOp, "filter", "NDArray.filter", &[ParamEntry::required("mask")]);
define_ndarray_stub!(NDArrayTolistOp, "tolist", "NDArray.tolist", &[]);
define_ndarray_stub!(NDArrayGetItemOp, "__get_item__", "NDArray.__get_item__", &[ParamEntry::required("key")]);
define_ndarray_stub!(NDArraySetItemOp, "__set_item__", "NDArray.__set_item__", &[ParamEntry::required("key"), ParamEntry::required("value")]);
define_ndarray_stub!(NDArrayRepeatOp, "repeat", "NDArray.repeat", &[ParamEntry::required("repeats"), ParamEntry::optional("axis")]);
