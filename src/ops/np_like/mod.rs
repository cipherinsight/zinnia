//! NumPy-like operators: np.add, np.subtract, np.multiply, np.divide, etc.
//! Most np_like ops are thin wrappers that delegate to the core arithmetic/math ops.
//! Ports `zinnia/op_def/np_like/` (74 files).
//!
//! Vectorization: when invoked on composite values (List/Tuple), the binary
//! and comparison ops fall through to `apply_binary_op` (which already does
//! shape-level broadcasting), and the unary-math ops walk leaves recursively.
//! Composite-aware behaviour is added by the macros below — individual op
//! files don't need to know about it.

/// Map an `np.<op>` name (as accepted by the np_like macros) to the short
/// op name expected by `helpers::value_ops::apply_binary_op`. Falls back on
/// the input string when the names already match.
pub(crate) fn np_op_name_to_apply_op(np_name: &str) -> &'static str {
    match np_name {
        "add" => "add",
        "subtract" => "sub",
        "multiply" => "mul",
        "divide" => "div",
        "floor_divide" => "floor_div",
        "mod" => "mod",
        "fmod" => "mod",
        "power" => "pow",
        "pow" => "pow",
        "equal" => "eq",
        "not_equal" => "ne",
        "less" => "lt",
        "less_equal" => "lte",
        "greater" => "gt",
        "greater_equal" => "gte",
        "logical_and" => "and",
        "logical_or" => "or",
        other => panic!("np_op_name_to_apply_op: no mapping for `{}`", other),
    }
}

/// Element-wise minimum/maximum that vectorizes over composites by
/// recursing into matching positions. Used by `define_np_minmax` when
/// either operand is a composite.
pub(crate) fn vectorize_minmax(
    builder: &mut crate::builder::IRBuilder,
    x1: &crate::types::Value,
    x2: &crate::types::Value,
    int_cmp: fn(&mut crate::builder::IRBuilder, &crate::types::Value, &crate::types::Value) -> crate::types::Value,
    float_cmp: fn(&mut crate::builder::IRBuilder, &crate::types::Value, &crate::types::Value) -> crate::types::Value,
) -> crate::types::Value {
    use crate::types::{CompositeData, Value};
    // Broadcast first so both operands have the same shape, then walk in
    // lockstep.
    let lshape = crate::helpers::composite::get_composite_shape(x1);
    let rshape = crate::helpers::composite::get_composite_shape(x2);
    let (l, r) = if lshape == rshape {
        (x1.clone(), x2.clone())
    } else if let Some(target) = crate::helpers::broadcast::broadcast_shapes(&lshape, &rshape) {
        (
            crate::helpers::broadcast::materialize_to_shape(x1, &target),
            crate::helpers::broadcast::materialize_to_shape(x2, &target),
        )
    } else {
        panic!(
            "minimum/maximum: shapes {:?} and {:?} are not broadcast compatible",
            lshape, rshape
        );
    };
    fn rec(
        builder: &mut crate::builder::IRBuilder,
        a: &Value,
        b: &Value,
        int_cmp: fn(&mut crate::builder::IRBuilder, &Value, &Value) -> Value,
        float_cmp: fn(&mut crate::builder::IRBuilder, &Value, &Value) -> Value,
    ) -> Value {
        match (a, b) {
            (Value::List(da), Value::List(db))
            | (Value::List(da), Value::Tuple(db))
            | (Value::Tuple(da), Value::List(db))
            | (Value::Tuple(da), Value::Tuple(db)) => {
                let vals: Vec<Value> = da
                    .values
                    .iter()
                    .zip(db.values.iter())
                    .map(|(av, bv)| rec(builder, av, bv, int_cmp, float_cmp))
                    .collect();
                let types = vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: types,
                    values: vals,
                })
            }
            _ => {
                let use_float = matches!(a, Value::Float(_)) || matches!(b, Value::Float(_));
                if use_float {
                    let af = if matches!(a, Value::Float(_)) {
                        a.clone()
                    } else {
                        builder.ir_float_cast(a)
                    };
                    let bf = if matches!(b, Value::Float(_)) {
                        b.clone()
                    } else {
                        builder.ir_float_cast(b)
                    };
                    let cond = float_cmp(builder, &af, &bf);
                    builder.ir_select_f(&cond, &af, &bf)
                } else {
                    let cond = int_cmp(builder, a, b);
                    builder.ir_select_i(&cond, a, b)
                }
            }
        }
    }
    rec(builder, &l, &r, int_cmp, float_cmp)
}

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
                // Vectorize over composites: fall through to apply_binary_op
                // which already handles broadcasting + element-wise dispatch.
                use crate::types::Value;
                if matches!(x1, Value::List(_) | Value::Tuple(_))
                    || matches!(x2, Value::List(_) | Value::Tuple(_))
                {
                    return crate::helpers::value_ops::apply_binary_op(
                        builder,
                        crate::ops::np_like::np_op_name_to_apply_op($op_name),
                        x1,
                        x2,
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
                use crate::types::Value;
                if matches!(x1, Value::List(_) | Value::Tuple(_))
                    || matches!(x2, Value::List(_) | Value::Tuple(_))
                {
                    return crate::helpers::value_ops::apply_binary_op(
                        builder,
                        crate::ops::np_like::np_op_name_to_apply_op($op_name),
                        x1,
                        x2,
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
        impl $name {
            // Recursively walk composites and apply the scalar op at the
            // leaves. Inlined inside the impl so the macro substitution can
            // reference the captured `$float_method` and `$sig`.
            fn apply_scalar(builder: &mut crate::builder::IRBuilder, x: &crate::types::Value) -> crate::types::Value {
                use crate::types::{Value, CompositeData};
                match x {
                    Value::List(d) | Value::Tuple(d) => {
                        let vals: Vec<Value> = d.values.iter()
                            .map(|v| Self::apply_scalar(builder, v))
                            .collect();
                        let types = vals.iter().map(|v| v.zinnia_type()).collect();
                        Value::List(CompositeData { elements_type: types, values: vals })
                    }
                    Value::Float(_) => builder.$float_method(x),
                    Value::Integer(_) | Value::Boolean(_) => {
                        let xf = builder.ir_float_cast(x);
                        builder.$float_method(&xf)
                    }
                    _ => panic!("{}: unsupported type {:?}", $sig, x.zinnia_type()),
                }
            }
        }
        impl crate::ops::Op for $name {
            fn name(&self) -> &'static str { $op_name }
            fn signature(&self) -> &'static str { $sig }
            fn params(&self) -> &[crate::ops::ParamEntry] { &Self::PARAMS }
            fn build(&self, builder: &mut crate::builder::IRBuilder, args: &crate::ops::OpArgs) -> crate::types::Value {
                Self::apply_scalar(builder, args.require("x"))
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
                    (a, b) if matches!(a, crate::types::Value::List(_) | crate::types::Value::Tuple(_))
                        || matches!(b, crate::types::Value::List(_) | crate::types::Value::Tuple(_)) =>
                    {
                        // Vectorize over composites by recursing through
                        // shape-broadcast leaves.
                        crate::ops::np_like::vectorize_minmax(
                            builder, a, b,
                            crate::builder::IRBuilder::$int_cmp,
                            crate::builder::IRBuilder::$float_cmp,
                        )
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
