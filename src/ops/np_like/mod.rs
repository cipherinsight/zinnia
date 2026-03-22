//! NumPy-like operators: np.add, np.subtract, np.multiply, np.divide, etc.
//! Most np_like ops are thin wrappers that delegate to the core arithmetic/math ops.
//! Ports `zinnia/op_def/np_like/` (74 files).

macro_rules! define_np_arith {
    ($name:ident, $op_name:expr, $sig:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [crate::ops::ParamEntry; 2] = [
                crate::ops::ParamEntry::required("x1"),
                crate::ops::ParamEntry::required("x2"),
            ];
        }
        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                if x1.is_number() && x2.is_number() {
                    return crate::ops::dispatch_binary_numeric(
                        builder, x1, x2,
                        crate::builder::IRBuilder::$int_method, crate::builder::IRBuilder::$float_method,
                    );
                }
                panic!("{}: unsupported types", $sig);
            }
        }
    };
}
pub(crate) use define_np_arith;

macro_rules! define_np_compare {
    ($name:ident, $op_name:expr, $sig:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [crate::ops::ParamEntry; 2] = [
                crate::ops::ParamEntry::required("x1"),
                crate::ops::ParamEntry::required("x2"),
            ];
        }
        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                if x1.is_number() && x2.is_number() {
                    return crate::ops::dispatch_binary_numeric(
                        builder, x1, x2,
                        crate::builder::IRBuilder::$int_method, crate::builder::IRBuilder::$float_method,
                    );
                }
                panic!("{}: unsupported types", $sig);
            }
        }
    };
}
pub(crate) use define_np_compare;

macro_rules! define_np_unary_math {
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
pub(crate) use define_np_unary_math;

macro_rules! define_np_minmax {
    ($name:ident, $op_name:expr, $sig:expr, $int_cmp:ident, $float_cmp:ident) => {
        pub struct $name;
        impl $name {
            const PARAMS: [crate::ops::ParamEntry; 2] = [
                crate::ops::ParamEntry::required("x1"),
                crate::ops::ParamEntry::required("x2"),
            ];
        }
        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                let x1 = args.require("x1");
                let x2 = args.require("x2");
                match (x1, x2) {
                    (crate::types::Value::Integer(_), crate::types::Value::Integer(_))
                    | (crate::types::Value::Boolean(_), crate::types::Value::Integer(_))
                    | (crate::types::Value::Integer(_), crate::types::Value::Boolean(_))
                    | (crate::types::Value::Boolean(_), crate::types::Value::Boolean(_)) => {
                        let cond = builder.$int_cmp(x1, x2);
                        builder.ir_select_i(&cond, x1, x2)
                    }
                    (crate::types::Value::Float(_), crate::types::Value::Float(_)) => {
                        let cond = builder.$float_cmp(x1, x2);
                        builder.ir_select_f(&cond, x1, x2)
                    }
                    (crate::types::Value::Integer(_) | crate::types::Value::Boolean(_), crate::types::Value::Float(_)) => {
                        let x1f = builder.ir_float_cast(x1);
                        let cond = builder.$float_cmp(&x1f, x2);
                        builder.ir_select_f(&cond, &x1f, x2)
                    }
                    (crate::types::Value::Float(_), crate::types::Value::Integer(_) | crate::types::Value::Boolean(_)) => {
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
pub(crate) use define_np_minmax;

pub mod np_add;
pub mod np_subtract;
pub mod np_multiply;
pub mod np_divide;
pub mod np_floor_divide;
pub mod np_mod;
pub mod np_fmod;
pub mod np_power;
pub mod np_pow;
pub mod np_minimum;
pub mod np_maximum;
pub mod np_fmin;
pub mod np_fmax;
pub mod np_equal;
pub mod np_not_equal;
pub mod np_less;
pub mod np_less_equal;
pub mod np_greater;
pub mod np_greater_equal;
pub mod np_sqrt;
pub mod np_exp;
pub mod np_log;
pub mod np_sin;
pub mod np_cos;
pub mod np_tan;
pub mod np_sinh;
pub mod np_cosh;
pub mod np_tanh;
pub mod np_abs;
pub mod np_absolute;
pub mod np_fabs;
pub mod np_sign;
pub mod np_negative;
pub mod np_positive;
pub mod np_logical_not;
pub mod np_logical_and;
pub mod np_logical_or;
pub mod np_logical_xor;

pub use np_add::NpAddOp;
pub use np_subtract::NpSubtractOp;
pub use np_multiply::NpMultiplyOp;
pub use np_divide::NpDivideOp;
pub use np_floor_divide::NpFloorDivideOp;
pub use np_mod::NpModOp;
pub use np_fmod::NpFModOp;
pub use np_power::NpPowerOp;
pub use np_pow::NpPowOp;
pub use np_minimum::NpMinimumOp;
pub use np_maximum::NpMaximumOp;
pub use np_fmin::NpFMinOp;
pub use np_fmax::NpFMaxOp;
pub use np_equal::NpEqualOp;
pub use np_not_equal::NpNotEqualOp;
pub use np_less::NpLessOp;
pub use np_less_equal::NpLessEqualOp;
pub use np_greater::NpGreaterOp;
pub use np_greater_equal::NpGreaterEqualOp;
pub use np_sqrt::NpSqrtOp;
pub use np_exp::NpExpOp;
pub use np_log::NpLogOp;
pub use np_sin::NpSinOp;
pub use np_cos::NpCosOp;
pub use np_tan::NpTanOp;
pub use np_sinh::NpSinHOp;
pub use np_cosh::NpCosHOp;
pub use np_tanh::NpTanHOp;
pub use np_abs::NpAbsOp;
pub use np_absolute::NpAbsoluteOp;
pub use np_fabs::NpFAbsOp;
pub use np_sign::NpSignOp;
pub use np_negative::NpNegativeOp;
pub use np_positive::NpPositiveOp;
pub use np_logical_not::NpLogicalNotOp;
pub use np_logical_and::NpLogicalAndOp;
pub use np_logical_or::NpLogicalOrOp;
pub use np_logical_xor::NpLogicalXorOp;
