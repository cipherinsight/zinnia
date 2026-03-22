//! NumPy-like operators: np.add, np.subtract, np.multiply, np.divide, etc.
//! Most np_like ops are thin wrappers that delegate to the core arithmetic/math ops.
//! Ports `zinnia/op_def/np_like/` (74 files).

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

macro_rules! define_np_arith {
    ($name:ident, $op_name:expr, $sig:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [ParamEntry; 2] = [
                ParamEntry::required("x1"),
                ParamEntry::required("x2"),
            ];
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                if x1.is_number() && x2.is_number() {
                    return crate::ops::arithmetic::binary_number_op_pub(
                        builder, x1, x2,
                        IRBuilder::$int_method, IRBuilder::$float_method,
                    );
                }
                panic!("{}: unsupported types", $sig);
            }
        }
    };
}

macro_rules! define_np_compare {
    ($name:ident, $op_name:expr, $sig:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [ParamEntry; 2] = [
                ParamEntry::required("x1"),
                ParamEntry::required("x2"),
            ];
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                if x1.is_number() && x2.is_number() {
                    return crate::ops::comparison::compare_number_op_pub(
                        builder, x1, x2,
                        IRBuilder::$int_method, IRBuilder::$float_method,
                    );
                }
                panic!("{}: unsupported types", $sig);
            }
        }
    };
}

macro_rules! define_np_unary_math {
    ($name:ident, $op_name:expr, $sig:expr, $float_method:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let x = args.require("x");
                match x {
                    Value::Float(_) => builder.$float_method(x),
                    Value::Integer(_) | Value::Boolean(_) => {
                        let xf = builder.ir_float_cast(x);
                        builder.$float_method(&xf)
                    }
                    _ => panic!("{}: unsupported type {:?}", $sig, x.zinnia_type()),
                }
            }
        }
    };
}

macro_rules! define_np_stub {
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

// ── Binary arithmetic ─────────────────────────────────────────────────
define_np_arith!(NpAddOp, "add", "np.add", ir_add_i, ir_add_f);
define_np_arith!(NpSubtractOp, "subtract", "np.subtract", ir_sub_i, ir_sub_f);
define_np_arith!(NpMultiplyOp, "multiply", "np.multiply", ir_mul_i, ir_mul_f);
define_np_arith!(NpDivideOp, "divide", "np.divide", ir_div_i, ir_div_f);
define_np_arith!(NpFloorDivideOp, "floor_divide", "np.floor_divide", ir_floor_div_i, ir_floor_div_f);
define_np_arith!(NpModOp, "mod", "np.mod", ir_mod_i, ir_mod_f);
define_np_arith!(NpFModOp, "fmod", "np.fmod", ir_mod_i, ir_mod_f);
define_np_arith!(NpPowerOp, "power", "np.power", ir_pow_i, ir_pow_f);
define_np_arith!(NpPowOp, "pow", "np.pow", ir_pow_i, ir_pow_f);
// np.minimum, np.maximum, np.fmax, np.fmin use select-based min/max logic.
// They cannot use the binary arithmetic macro since they need comparison + select.

macro_rules! define_np_minmax {
    ($name:ident, $op_name:expr, $sig:expr, $int_cmp:ident, $float_cmp:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [ParamEntry; 2] = [
                ParamEntry::required("x1"),
                ParamEntry::required("x2"),
            ];
        }
        impl Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                match (x1, x2) {
                    (Value::Integer(_), Value::Integer(_))
                    | (Value::Boolean(_), Value::Integer(_))
                    | (Value::Integer(_), Value::Boolean(_))
                    | (Value::Boolean(_), Value::Boolean(_)) => {
                        let cond = builder.$int_cmp(x1, x2);
                        builder.ir_select_i(&cond, x1, x2)
                    }
                    (Value::Float(_), Value::Float(_)) => {
                        let cond = builder.$float_cmp(x1, x2);
                        builder.ir_select_f(&cond, x1, x2)
                    }
                    (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
                        let x1f = builder.ir_float_cast(x1);
                        let cond = builder.$float_cmp(&x1f, x2);
                        builder.ir_select_f(&cond, &x1f, x2)
                    }
                    (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
                        let x2f = builder.ir_float_cast(x2);
                        let cond = builder.$float_cmp(x1, &x2f);
                        builder.ir_select_f(&cond, x1, &x2f)
                    }
                    _ => panic!("{}: unsupported types", $sig),
                }
            }
        }
    };
}

define_np_minmax!(NpMinimumOp, "minimum", "np.minimum", ir_less_than_i, ir_less_than_f);
define_np_minmax!(NpMaximumOp, "maximum", "np.maximum", ir_greater_than_i, ir_greater_than_f);
define_np_minmax!(NpFMinOp, "fmin", "np.fmin", ir_less_than_i, ir_less_than_f);
define_np_minmax!(NpFMaxOp, "fmax", "np.fmax", ir_greater_than_i, ir_greater_than_f);

// ── Comparisons ───────────────────────────────────────────────────────
define_np_compare!(NpEqualOp, "equal", "np.equal", ir_equal_i, ir_equal_f);
define_np_compare!(NpNotEqualOp, "not_equal", "np.not_equal", ir_not_equal_i, ir_not_equal_f);
define_np_compare!(NpLessOp, "less", "np.less", ir_less_than_i, ir_less_than_f);
define_np_compare!(NpLessEqualOp, "less_equal", "np.less_equal", ir_less_than_or_equal_i, ir_less_than_or_equal_f);
define_np_compare!(NpGreaterOp, "greater", "np.greater", ir_greater_than_i, ir_greater_than_f);
define_np_compare!(NpGreaterEqualOp, "greater_equal", "np.greater_equal", ir_greater_than_or_equal_i, ir_greater_than_or_equal_f);

// ── Unary math ────────────────────────────────────────────────────────
define_np_unary_math!(NpSqrtOp, "sqrt", "np.sqrt", ir_sqrt_f);
define_np_unary_math!(NpExpOp, "exp", "np.exp", ir_exp_f);
define_np_unary_math!(NpLogOp, "log", "np.log", ir_log_f);
define_np_unary_math!(NpSinOp, "sin", "np.sin", ir_sin_f); // asin placeholder
define_np_unary_math!(NpCosOp, "cos", "np.cos", ir_cos_f);
define_np_unary_math!(NpTanOp, "tan", "np.tan", ir_tan_f);
define_np_unary_math!(NpSinHOp, "sinh", "np.sinh", ir_sinh_f);
define_np_unary_math!(NpCosHOp, "cosh", "np.cosh", ir_cosh_f);
define_np_unary_math!(NpTanHOp, "tanh", "np.tanh", ir_tanh_f);
define_np_unary_math!(NpAbsOp, "abs", "np.abs", ir_abs_f);
define_np_unary_math!(NpAbsoluteOp, "absolute", "np.absolute", ir_abs_f);
define_np_unary_math!(NpFAbsOp, "fabs", "np.fabs", ir_abs_f);
// np.acos, np.asin, np.atan are NOT yet supported — no inverse trig IR instructions exist.
define_np_stub!(NpACosOp, "acos", "np.acos", &[ParamEntry::required("x")]);
define_np_stub!(NpASinOp, "asin", "np.asin", &[ParamEntry::required("x")]);
define_np_stub!(NpATanOp, "atan", "np.atan", &[ParamEntry::required("x")]);

// np.sign needs to handle both int and float inputs.
pub struct NpSignOp;
impl NpSignOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for NpSignOp {
    fn name(&self) -> &'static str { "sign" }
    fn signature(&self) -> &'static str { "np.sign" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_sign_i(x),
            Value::Float(_) => builder.ir_sign_f(x),
            _ => panic!("np.sign: unsupported type {:?}", x.zinnia_type()),
        }
    }
}

// ── Unary ops ─────────────────────────────────────────────────────────

pub struct NpNegativeOp;
impl NpNegativeOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for NpNegativeOp {
    fn name(&self) -> &'static str { "negative" }
    fn signature(&self) -> &'static str { "np.negative" }
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
            _ => panic!("np.negative: unsupported type"),
        }
    }
}

pub struct NpPositiveOp;
impl NpPositiveOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for NpPositiveOp {
    fn name(&self) -> &'static str { "positive" }
    fn signature(&self) -> &'static str { "np.positive" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, _builder: &mut IRBuilder, args: &OpArgs) -> Value {
        args.require("x").clone()
    }
}

// ── Logical ops ───────────────────────────────────────────────────────

pub struct NpLogicalNotOp;
impl NpLogicalNotOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}
impl Op for NpLogicalNotOp {
    fn name(&self) -> &'static str { "logical_not" }
    fn signature(&self) -> &'static str { "np.logical_not" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Boolean(_) => builder.ir_logical_not(x),
            Value::Integer(_) => {
                let b = builder.ir_bool_cast(x);
                builder.ir_logical_not(&b)
            }
            _ => panic!("np.logical_not: unsupported type"),
        }
    }
}

pub struct NpLogicalAndOp;
impl NpLogicalAndOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("x1"),
        ParamEntry::required("x2"),
    ];
}
impl Op for NpLogicalAndOp {
    fn name(&self) -> &'static str { "logical_and" }
    fn signature(&self) -> &'static str { "np.logical_and" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        builder.ir_logical_and(x1, x2)
    }
}

pub struct NpLogicalOrOp;
impl NpLogicalOrOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("x1"),
        ParamEntry::required("x2"),
    ];
}
impl Op for NpLogicalOrOp {
    fn name(&self) -> &'static str { "logical_or" }
    fn signature(&self) -> &'static str { "np.logical_or" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        builder.ir_logical_or(x1, x2)
    }
}

pub struct NpLogicalXorOp;
impl NpLogicalXorOp {
    const PARAMS: [ParamEntry; 2] = [
        ParamEntry::required("x1"),
        ParamEntry::required("x2"),
    ];
}
impl Op for NpLogicalXorOp {
    fn name(&self) -> &'static str { "logical_xor" }
    fn signature(&self) -> &'static str { "np.logical_xor" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x1 = args.require("x1");
        let x2 = args.require("x2");
        let or_val = builder.ir_logical_or(x1, x2);
        let and_val = builder.ir_logical_and(x1, x2);
        let not_and = builder.ir_logical_not(&and_val);
        builder.ir_logical_and(&or_val, &not_and)
    }
}

// ── Stub ops for array creation, reduction, etc. ──────────────────────
// These require full NDArray value manipulation and will be completed
// when NDArray methods are fully implemented.

define_np_stub!(NpZerosOp, "zeros", "np.zeros", &[ParamEntry::required("shape"), ParamEntry::optional("dtype")]);
define_np_stub!(NpOnesOp, "ones", "np.ones", &[ParamEntry::required("shape"), ParamEntry::optional("dtype")]);
define_np_stub!(NpEyeOp, "eye", "np.eye", &[ParamEntry::required("N"), ParamEntry::optional("M"), ParamEntry::optional("dtype")]);
define_np_stub!(NpIdentityOp, "identity", "np.identity", &[ParamEntry::required("n"), ParamEntry::optional("dtype")]);
define_np_stub!(NpConcatenateOp, "concatenate", "np.concatenate", &[ParamEntry::required("arrays"), ParamEntry::optional("axis")]);
define_np_stub!(NpConcatOp, "concat", "np.concat", &[ParamEntry::required("arrays"), ParamEntry::optional("axis")]);
define_np_stub!(NpStackOp, "stack", "np.stack", &[ParamEntry::required("arrays"), ParamEntry::optional("axis")]);
define_np_stub!(NpAsarrayOp, "asarray", "np.asarray", &[ParamEntry::required("a")]);
define_np_stub!(NpArrayOp, "array", "np.array", &[ParamEntry::required("object"), ParamEntry::optional("dtype")]);
define_np_stub!(NpAllOp, "all", "np.all", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpAnyOp, "any", "np.any", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpAllCloseOp, "allclose", "np.allclose", &[ParamEntry::required("a"), ParamEntry::required("b")]);
define_np_stub!(NpIsCloseOp, "isclose", "np.isclose", &[ParamEntry::required("a"), ParamEntry::required("b")]);
define_np_stub!(NpArrayEqualOp, "array_equal", "np.array_equal", &[ParamEntry::required("a1"), ParamEntry::required("a2")]);
define_np_stub!(NpArrayEquivOp, "array_equiv", "np.array_equiv", &[ParamEntry::required("a1"), ParamEntry::required("a2")]);
define_np_stub!(NpArgmaxOp, "argmax", "np.argmax", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpArgminOp, "argmin", "np.argmin", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpAMaxOp, "amax", "np.amax", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpAMinOp, "amin", "np.amin", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpMaxOp, "max", "np.max", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpSumOp, "sum", "np.sum", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpProdOp, "prod", "np.prod", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpRepeatOp, "repeat", "np.repeat", &[ParamEntry::required("a"), ParamEntry::required("repeats")]);
define_np_stub!(NpSizeOp, "size", "np.size", &[ParamEntry::required("a")]);
define_np_stub!(NpDotOp, "dot", "np.dot", &[ParamEntry::required("a"), ParamEntry::required("b")]);
define_np_stub!(NpAppendOp, "append", "np.append", &[ParamEntry::required("arr"), ParamEntry::required("values")]);
define_np_stub!(NpARangeOp, "arange", "np.arange", &[ParamEntry::required("start"), ParamEntry::optional("stop"), ParamEntry::optional("step")]);
define_np_stub!(NpLinspaceOp, "linspace", "np.linspace", &[ParamEntry::required("start"), ParamEntry::required("stop"), ParamEntry::optional("num")]);
define_np_stub!(NpMeanOp, "mean", "np.mean", &[ParamEntry::required("a"), ParamEntry::optional("axis")]);
define_np_stub!(NpMoveAxisOp, "moveaxis", "np.moveaxis", &[ParamEntry::required("a"), ParamEntry::required("source"), ParamEntry::required("destination")]);
define_np_stub!(NpTransposeOp, "transpose", "np.transpose", &[ParamEntry::required("a"), ParamEntry::optional("axes")]);
