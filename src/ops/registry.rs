//! Operator registry: builds the NumPy-like scalar operators by name. `ir_gen` dispatches builtins and array, list and tuple methods
//! itself and falls back to this registry for `np.*` / `math.*` calls
//! (`ir_gen::named_attr::numpy_funcs::try_np_fallback`).

use crate::builder::IRBuilder;
use crate::ops::*;
use crate::types::Value;

/// Look up and build an operator by name. Returns `None` if the operator is
/// not found.
pub fn build_op(op_name: &str, builder: &mut IRBuilder, args: &OpArgs) -> Option<Value> {
    build_np_op(op_name, builder, args)
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
        "arccos" | "acos" => np_like::NpArcCosOp.build(builder, args),
        "arcsin" | "asin" => np_like::NpArcSinOp.build(builder, args),
        "arctan2" | "atan2" => np_like::NpArcTan2Op.build(builder, args),
        "abs" => np_like::NpAbsOp.build(builder, args),
        "absolute" => np_like::NpAbsoluteOp.build(builder, args),
        "fabs" => np_like::NpFAbsOp.build(builder, args),
        "inv" => math_ops::MathInvOp.build(builder, args),
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
        "heaviside" => np_like::NpHeavisideOp.build(builder, args),
        // Remaining np ops are dispatched directly by ir_gen
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_registry_np_add() {
        let mut b = IRBuilder::new();
        let a = b.ir_constant_int(10);
        let c = b.ir_constant_int(20);
        let mut kw = HashMap::new();
        kw.insert("x1".to_string(), a);
        kw.insert("x2".to_string(), c);
        let args = OpArgs::new(kw);
        let result = build_op("add", &mut b, &args).unwrap();
        assert_eq!(result.int_val(), Some(30));
    }
}
