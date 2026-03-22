//! Comparison operators: eq, ne, lt, lte, gt, gte.
//! Ports `zinnia/op_def/arithmetic/op_eq.py`, `op_ne.py`, `op_lt.py`, etc.

macro_rules! define_compare_op {
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

                if lhs.is_number() && rhs.is_number() {
                    return crate::ops::dispatch_binary_numeric(
                        builder,
                        lhs,
                        rhs,
                        crate::builder::IRBuilder::$int_method,
                        crate::builder::IRBuilder::$float_method,
                    );
                }

                // TODO: NDArray comparison
                panic!("Op `{}` not supported for {:?} and {:?}",
                    $op_name, lhs.zinnia_type(), rhs.zinnia_type())
            }
        }
    };
}
pub(crate) use define_compare_op;

pub mod equal;
pub mod not_equal;
pub mod less_than;
pub mod less_than_or_equal;
pub mod greater_than;
pub mod greater_than_or_equal;

pub use equal::EqualOp;
pub use not_equal::NotEqualOp;
pub use less_than::LessThanOp;
pub use less_than_or_equal::LessThanOrEqualOp;
pub use greater_than::GreaterThanOp;
pub use greater_than_or_equal::GreaterThanOrEqualOp;
