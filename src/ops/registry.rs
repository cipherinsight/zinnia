//! Operator registry — mirrors Python `OperatorFactory`.
//! Maps (operator_name, namespace) to Op instances.

use crate::builder::IRBuilder;
use crate::ops::*;
use crate::types::Value;

/// Namespace for operator lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpNamespace {
    NoCls,      // Global operators (no class prefix)
    NDArray,
    DynamicNDArray,
    Tuple,
    List,
    String,
    Np,         // numpy-like
    Zinnia,     // alias for Np
    Math,
}

impl OpNamespace {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "NDArray" => Some(Self::NDArray),
            "DynamicNDArray" => Some(Self::DynamicNDArray),
            "Tuple" => Some(Self::Tuple),
            "List" => Some(Self::List),
            "String" => Some(Self::String),
            "np" => Some(Self::Np),
            "zinnia" => Some(Self::Zinnia),
            "math" => Some(Self::Math),
            _ => None,
        }
    }

    pub fn namespaces() -> &'static [&'static str] {
        &["NDArray", "Tuple", "List", "String", "np", "zinnia", "math"]
    }
}

/// Look up and build an operator by name and optional namespace.
/// Returns None if the operator is not found.
/// This replaces the Python `Operators.get_operator()` + `op.build()` pattern.
pub fn build_op(
    op_name: &str,
    namespace: Option<OpNamespace>,
    builder: &mut IRBuilder,
    args: &OpArgs,
) -> Option<Value> {
    match namespace.unwrap_or(OpNamespace::NoCls) {
        OpNamespace::NoCls => build_nocls_op(op_name, builder, args),
        OpNamespace::Math => build_math_op(op_name, builder, args),
        OpNamespace::Np | OpNamespace::Zinnia => build_np_op(op_name, builder, args),
        OpNamespace::NDArray => build_ndarray_op(op_name, builder, args),
        OpNamespace::List => build_list_op(op_name, builder, args),
        OpNamespace::Tuple => build_tuple_op(op_name, builder, args),
        OpNamespace::DynamicNDArray => build_dynamic_ndarray_op(op_name, builder, args),
        OpNamespace::String => None, // String methods not yet implemented
    }
}

fn build_nocls_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    let result = match name {
        "tuple" => nocls::TupleOp.build(builder, args),
        "str" => cast::StrOp.build(builder, args),
        "range" => nocls::RangeOp.build(builder, args),
        "print" => nocls::PrintOp.build(builder, args),
        "pow" => nocls::PowOp.build(builder, args),
        "min" => nocls::MinOp.build(builder, args),
        "max" => nocls::MaxOp.build(builder, args),
        "list" => nocls::ListOp.build(builder, args),
        "len" => nocls::LenOp.build(builder, args),
        "float" => cast::FloatCastOp.build(builder, args),
        "bool" => cast::BoolCastOp.build(builder, args),
        "int" => cast::IntCastOp.build(builder, args),
        "sum" => nocls::SumOp.build(builder, args),
        "any" => nocls::AnyOp.build(builder, args),
        "all" => nocls::AllOp.build(builder, args),
        "abs" => nocls::AbsOp.build(builder, args),
        "poseidon_hash" => nocls::PoseidonHashBuiltinOp.build(builder, args),
        "merkle_verify" => nocls::MerkleVerifyOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_math_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    let result = match name {
        "sin" => math_ops::MathSinOp.build(builder, args),
        "cos" => math_ops::MathCosOp.build(builder, args),
        "tan" => math_ops::MathTanOp.build(builder, args),
        "sinh" => math_ops::MathSinHOp.build(builder, args),
        "cosh" => math_ops::MathCosHOp.build(builder, args),
        "tanh" => math_ops::MathTanHOp.build(builder, args),
        "sqrt" => math_ops::MathSqrtOp.build(builder, args),
        "exp" => math_ops::MathExpOp.build(builder, args),
        "log" => math_ops::MathLogOp.build(builder, args),
        "fabs" => math_ops::MathFAbsOp.build(builder, args),
        "inv" => math_ops::MathInvOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_np_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    let result = match name {
        "add" => np_like::NpAddOp.build(builder, args),
        "subtract" => np_like::NpSubtractOp.build(builder, args),
        "multiply" => np_like::NpMultiplyOp.build(builder, args),
        "divide" => np_like::NpDivideOp.build(builder, args),
        "floor_divide" => np_like::NpFloorDivideOp.build(builder, args),
        "mod" => np_like::NpModOp.build(builder, args),
        "fmod" => np_like::NpFModOp.build(builder, args),
        "power" => np_like::NpPowerOp.build(builder, args),
        "pow" => np_like::NpPowOp.build(builder, args),
        "equal" => np_like::NpEqualOp.build(builder, args),
        "not_equal" => np_like::NpNotEqualOp.build(builder, args),
        "less" => np_like::NpLessOp.build(builder, args),
        "less_equal" => np_like::NpLessEqualOp.build(builder, args),
        "greater" => np_like::NpGreaterOp.build(builder, args),
        "greater_equal" => np_like::NpGreaterEqualOp.build(builder, args),
        "sqrt" => np_like::NpSqrtOp.build(builder, args),
        "exp" => np_like::NpExpOp.build(builder, args),
        "log" => np_like::NpLogOp.build(builder, args),
        "sin" => np_like::NpSinOp.build(builder, args),
        "cos" => np_like::NpCosOp.build(builder, args),
        "tan" => np_like::NpTanOp.build(builder, args),
        "sinh" => np_like::NpSinHOp.build(builder, args),
        "cosh" => np_like::NpCosHOp.build(builder, args),
        "tanh" => np_like::NpTanHOp.build(builder, args),
        "abs" => np_like::NpAbsOp.build(builder, args),
        "absolute" => np_like::NpAbsoluteOp.build(builder, args),
        "fabs" => np_like::NpFAbsOp.build(builder, args),
        "sign" => np_like::NpSignOp.build(builder, args),
        "negative" => np_like::NpNegativeOp.build(builder, args),
        "positive" => np_like::NpPositiveOp.build(builder, args),
        "logical_not" => np_like::NpLogicalNotOp.build(builder, args),
        "logical_and" => np_like::NpLogicalAndOp.build(builder, args),
        "logical_or" => np_like::NpLogicalOrOp.build(builder, args),
        "logical_xor" => np_like::NpLogicalXorOp.build(builder, args),
        "minimum" => np_like::NpMinimumOp.build(builder, args),
        "maximum" => np_like::NpMaximumOp.build(builder, args),
        "fmin" => np_like::NpFMinOp.build(builder, args),
        "fmax" => np_like::NpFMaxOp.build(builder, args),
        "acos" => np_like::NpACosOp.build(builder, args),
        "asin" => np_like::NpASinOp.build(builder, args),
        "atan" => np_like::NpATanOp.build(builder, args),
        // Array creation & reduction stubs (panic with clear message if invoked)
        "zeros" => np_like::NpZerosOp.build(builder, args),
        "ones" => np_like::NpOnesOp.build(builder, args),
        "eye" => np_like::NpEyeOp.build(builder, args),
        "identity" => np_like::NpIdentityOp.build(builder, args),
        "concatenate" => np_like::NpConcatenateOp.build(builder, args),
        "concat" => np_like::NpConcatOp.build(builder, args),
        "stack" => np_like::NpStackOp.build(builder, args),
        "asarray" => np_like::NpAsarrayOp.build(builder, args),
        "array" => np_like::NpArrayOp.build(builder, args),
        "all" => np_like::NpAllOp.build(builder, args),
        "any" => np_like::NpAnyOp.build(builder, args),
        "allclose" => np_like::NpAllCloseOp.build(builder, args),
        "isclose" => np_like::NpIsCloseOp.build(builder, args),
        "array_equal" => np_like::NpArrayEqualOp.build(builder, args),
        "array_equiv" => np_like::NpArrayEquivOp.build(builder, args),
        "argmax" => np_like::NpArgmaxOp.build(builder, args),
        "argmin" => np_like::NpArgminOp.build(builder, args),
        "amax" => np_like::NpAMaxOp.build(builder, args),
        "amin" => np_like::NpAMinOp.build(builder, args),
        "max" => np_like::NpMaxOp.build(builder, args),
        "sum" => np_like::NpSumOp.build(builder, args),
        "prod" => np_like::NpProdOp.build(builder, args),
        "repeat" => np_like::NpRepeatOp.build(builder, args),
        "size" => np_like::NpSizeOp.build(builder, args),
        "dot" => np_like::NpDotOp.build(builder, args),
        "append" => np_like::NpAppendOp.build(builder, args),
        "arange" => np_like::NpARangeOp.build(builder, args),
        "linspace" => np_like::NpLinspaceOp.build(builder, args),
        "mean" => np_like::NpMeanOp.build(builder, args),
        "moveaxis" => np_like::NpMoveAxisOp.build(builder, args),
        "transpose" => np_like::NpTransposeOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_ndarray_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    use crate::ops::ndarray_ops::*;
    let result = match name {
        "sum" => NDArraySumOp.build(builder, args),
        "prod" => NDArrayProdOp.build(builder, args),
        "max" => NDArrayMaxOp.build(builder, args),
        "min" => NDArrayMinOp.build(builder, args),
        "argmax" => NDArrayArgmaxOp.build(builder, args),
        "argmin" => NDArrayArgminOp.build(builder, args),
        "all" => NDArrayAllOp.build(builder, args),
        "any" => NDArrayAnyOp.build(builder, args),
        "reshape" => NDArrayReshapeOp.build(builder, args),
        "flatten" => NDArrayFlattenOp.build(builder, args),
        "T" => NDArrayTOp.build(builder, args),
        "transpose" => NDArrayTransposeOp.build(builder, args),
        "moveaxis" => NDArrayMoveaxisOp.build(builder, args),
        "astype" => NDArrayAstypeOp.build(builder, args),
        "dtype" => NDArrayDtypeOp.build(builder, args),
        "shape" => NDArrayShapeOp.build(builder, args),
        "size" => NDArraySizeOp.build(builder, args),
        "ndim" => NDArrayNdimOp.build(builder, args),
        "flat" => NDArrayFlatOp.build(builder, args),
        "filter" => NDArrayFilterOp.build(builder, args),
        "tolist" => NDArrayTolistOp.build(builder, args),
        "__get_item__" => NDArrayGetItemOp.build(builder, args),
        "__set_item__" => NDArraySetItemOp.build(builder, args),
        "repeat" => NDArrayRepeatOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_list_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    use crate::ops::list_ops::*;
    let result = match name {
        "append" => ListAppendOp.build(builder, args),
        "extend" => ListExtendOp.build(builder, args),
        "insert" => ListInsertOp.build(builder, args),
        "pop" => ListPopOp.build(builder, args),
        "remove" => ListRemoveOp.build(builder, args),
        "clear" => ListClearOp.build(builder, args),
        "index" => ListIndexOp.build(builder, args),
        "count" => ListCountOp.build(builder, args),
        "reverse" => ListReverseOp.build(builder, args),
        "copy" => ListCopyOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_tuple_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    use crate::ops::tuple_ops::*;
    let result = match name {
        "count" => TupleCountOp.build(builder, args),
        "index" => TupleIndexOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

fn build_dynamic_ndarray_op(name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    use crate::ops::dynamic_ndarray_ops::*;
    let result = match name {
        "sum" => DynNDArraySumOp.build(builder, args),
        "prod" => DynNDArrayProdOp.build(builder, args),
        "max" => DynNDArrayMaxOp.build(builder, args),
        "min" => DynNDArrayMinOp.build(builder, args),
        "argmax" => DynNDArrayArgmaxOp.build(builder, args),
        "argmin" => DynNDArrayArgminOp.build(builder, args),
        "all" => DynNDArrayAllOp.build(builder, args),
        "any" => DynNDArrayAnyOp.build(builder, args),
        "flatten" => DynNDArrayFlattenOp.build(builder, args),
        "T" => DynNDArrayTOp.build(builder, args),
        "transpose" => DynNDArrayTransposeOp.build(builder, args),
        "moveaxis" => DynNDArrayMoveaxisOp.build(builder, args),
        "astype" => DynNDArrayAstypeOp.build(builder, args),
        "dtype" => DynNDArrayDtypeOp.build(builder, args),
        "shape" => DynNDArrayShapeOp.build(builder, args),
        "size" => DynNDArraySizeOp.build(builder, args),
        "ndim" => DynNDArrayNdimOp.build(builder, args),
        "flat" => DynNDArrayFlatOp.build(builder, args),
        "filter" => DynNDArrayFilterOp.build(builder, args),
        "tolist" => DynNDArrayTolistOp.build(builder, args),
        "__get_item__" => DynNDArrayGetItemOp.build(builder, args),
        "__set_item__" => DynNDArraySetItemOp.build(builder, args),
        "repeat" => DynNDArrayRepeatOp.build(builder, args),
        "concatenate" => DynNDArrayConcatenateOp.build(builder, args),
        "stack" => DynNDArrayStackOp.build(builder, args),
        "zeros" => DynNDArrayZerosOp.build(builder, args),
        "ones" => DynNDArrayOnesOp.build(builder, args),
        "eye" => DynNDArrayEyeOp.build(builder, args),
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_registry_nocls_int() {
        let mut b = IRBuilder::new();
        let x = b.ir_constant_float(3.14);
        let mut kw = HashMap::new();
        kw.insert("x".to_string(), x);
        let args = OpArgs::new(kw);
        let result = build_op("int", None, &mut b, &args).unwrap();
        assert_eq!(result.int_val(), Some(3));
    }

    #[test]
    fn test_registry_math_sqrt() {
        let mut b = IRBuilder::new();
        let x = b.ir_constant_float(4.0);
        let mut kw = HashMap::new();
        kw.insert("x".to_string(), x);
        let args = OpArgs::new(kw);
        let result = build_op("sqrt", Some(OpNamespace::Math), &mut b, &args).unwrap();
        assert_eq!(result.float_val(), Some(2.0));
    }

    #[test]
    fn test_registry_np_add() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(10);
        let c = b.ir_constant_int(20);
        let mut kw = HashMap::new();
        kw.insert("x1".to_string(), a);
        kw.insert("x2".to_string(), c);
        let args = OpArgs::new(kw);
        let result = build_op("add", Some(OpNamespace::Np), &mut b, &args).unwrap();
        assert_eq!(result.int_val(), Some(30));
    }
}
