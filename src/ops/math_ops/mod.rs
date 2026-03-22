//! Math function operators: sin, cos, tan, sinh, cosh, tanh, sqrt, exp, log, fabs, inv.
//! Ports `zinnia/op_def/math/`.

macro_rules! define_math_unary_op {
    ($name:ident, $op_name:expr, $sig:expr, $float_method:ident) => {
        pub struct $name;

        impl $name {
            const PARAMS: [crate::ops::ParamEntry; 1] = [crate::ops::ParamEntry::required("x")];
        }

        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }

            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                let x = args.require("x");
                match x {
                    crate::types::Value::Float(_) => builder.$float_method(x),
                    crate::types::Value::Integer(_) | crate::types::Value::Boolean(_) => {
                        let xf = builder.ir_float_cast(x);
                        builder.$float_method(&xf)
                    }
                    _ => panic!("{}: unsupported type {:?}", $sig, x.zinnia_type()),
                }
            }
        }
    };
}
pub(crate) use define_math_unary_op;

pub mod math_sin;
pub mod math_cos;
pub mod math_tan;
pub mod math_sinh;
pub mod math_cosh;
pub mod math_tanh;
pub mod math_sqrt;
pub mod math_exp;
pub mod math_log;
pub mod math_fabs;
pub mod math_inv;

pub use math_sin::MathSinOp;
pub use math_cos::MathCosOp;
pub use math_tan::MathTanOp;
pub use math_sinh::MathSinHOp;
pub use math_cosh::MathCosHOp;
pub use math_tanh::MathTanHOp;
pub use math_sqrt::MathSqrtOp;
pub use math_exp::MathExpOp;
pub use math_log::MathLogOp;
pub use math_fabs::MathFAbsOp;
pub use math_inv::MathInvOp;
