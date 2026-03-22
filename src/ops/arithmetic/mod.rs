//! Arithmetic operators: add, sub, mul, div, floor_divide, mod, power, usub, uadd, mat_mul.
//! Ports `zinnia/op_def/arithmetic/`.

macro_rules! define_binary_arith_op {
    ($name:ident, $op_name:expr, $int_method:ident, $float_method:ident) => {
        pub struct $name;

        impl $name {
            const PARAMS: [crate::ops::ParamEntry; 2] = [
                crate::ops::ParamEntry::required("lhs"),
                crate::ops::ParamEntry::required("rhs"),
            ];
        }

        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }

            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                let lhs = args.require("lhs");
                let rhs = args.require("rhs");

                // Scalar path
                if lhs.is_number() && rhs.is_number() {
                    return crate::ops::dispatch_binary_numeric(
                        builder,
                        lhs,
                        rhs,
                        crate::builder::IRBuilder::$int_method,
                        crate::builder::IRBuilder::$float_method,
                    );
                }

                // TODO: NDArray, List, Tuple, String paths
                panic!("Op `{}` not supported for {:?} and {:?}",
                    $op_name, lhs.zinnia_type(), rhs.zinnia_type())
            }
        }
    };
}
pub(crate) use define_binary_arith_op;

pub mod add;
pub mod sub;
pub mod mul;
pub mod div;
pub mod floor_divide;
pub mod mod_op;
pub mod power;
pub mod mat_mul;
pub mod usub;
pub mod uadd;

pub use add::AddOp;
pub use sub::SubOp;
pub use mul::MulOp;
pub use div::DivOp;
pub use floor_divide::FloorDivideOp;
pub use mod_op::ModOp;
pub use power::PowerOp;
pub use mat_mul::MatMulOp;
pub use usub::USubOp;
pub use uadd::UAddOp;
