//! Math function operators: sin, cos, tan, sinh, cosh, tanh, sqrt, exp, log, fabs, inv.
//! Ports `zinnia/op_def/math/`.

use crate::builder::IRBuilder;
use crate::ops::{Op, OpArgs, ParamEntry};
use crate::types::Value;

macro_rules! define_math_unary_op {
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

define_math_unary_op!(MathSinOp, "sin", "math.sin", ir_sin_f);
define_math_unary_op!(MathCosOp, "cos", "math.cos", ir_cos_f);
define_math_unary_op!(MathTanOp, "tan", "math.tan", ir_tan_f);
define_math_unary_op!(MathSinHOp, "sinh", "math.sinh", ir_sinh_f);
define_math_unary_op!(MathCosHOp, "cosh", "math.cosh", ir_cosh_f);
define_math_unary_op!(MathTanHOp, "tanh", "math.tanh", ir_tanh_f);
define_math_unary_op!(MathSqrtOp, "sqrt", "math.sqrt", ir_sqrt_f);
define_math_unary_op!(MathExpOp, "exp", "math.exp", ir_exp_f);
define_math_unary_op!(MathLogOp, "log", "math.log", ir_log_f);
define_math_unary_op!(MathFAbsOp, "fabs", "math.fabs", ir_abs_f);

pub struct MathInvOp;

impl MathInvOp {
    const PARAMS: [ParamEntry; 1] = [ParamEntry::required("x")];
}

impl Op for MathInvOp {
    fn name(&self) -> &'static str { "inv" }
    fn signature(&self) -> &'static str { "math.inv" }
    fn params(&self) -> &[ParamEntry] { &Self::PARAMS }

    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value {
        let x = args.require("x");
        match x {
            Value::Integer(_) | Value::Boolean(_) => builder.ir_inv_i(x),
            _ => panic!("math.inv: unsupported type {:?}", x.zinnia_type()),
        }
    }
}
