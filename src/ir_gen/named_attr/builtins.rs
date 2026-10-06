//! Python-builtin dispatch arms (no target on the named-attribute call):
//! `range`, `len`, `int`, `float`, `bool`, `complex`, `abs`, `print`,
//! `enumerate`, `sum`, `any`, `all`, `min`, `max`, `pow`, `list`, `tuple`.
//!
//! Returns `None` when the (target, member) pair is not a recognized builtin,
//! so `visit_named_attr` can fall through to the next category.

use crate::types::{CompositeData, Value, ValueId};

use super::super::IRGenerator;
use super::DispatchCtx;

impl IRGenerator {
    pub(crate) fn try_builtin(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        if ctx.target.is_some() {
            return None;
        }
        let visited_args = ctx.args;
        let visited_args_orig = ctx.args_orig;
        let member = ctx.member;
        let v = match member {
            "zip" => {
                let sequences: Vec<Vec<Value>> = visited_args.iter().map(|v| match v {
                    Value::List(d) | Value::Tuple(d) => d.values.clone(),
                    _ => panic!("zip: expected statically bounded iterables"),
                }).collect();
                let n = sequences.iter().map(Vec::len).min().unwrap_or(0);
                if super::kwarg_static_truthy(ctx.kwargs, "strict", false) {
                    assert!(sequences.iter().all(|v| v.len() == n), "zip: argument lengths differ with strict=True");
                }
                let values: Vec<Value> = (0..n).map(|i| {
                    let row: Vec<_> = sequences.iter().map(|v| v[i].clone()).collect();
                    Value::Tuple(CompositeData::new(row.iter().map(Value::zinnia_type).collect(), row))
                }).collect();
                Value::Tuple(CompositeData::new(values.iter().map(Value::zinnia_type).collect(), values))
            }
            "range" => {
                crate::ops::static_ndarray_ops::builtin_range(&mut self.builder, visited_args)
            }
            "len" => {
                if let Some(Value::DynamicNDArray(data)) = visited_args_orig.first() {
                    self.note_len_bounds(data);
                }
                crate::ops::static_ndarray_ops::builtin_len(&mut self.builder, visited_args)
            }
            "int" => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_int(0)
                } else {
                    reject_complex_conversion("int", &visited_args[0]);
                    // A Python int: an integer cell (a bool's indexes by
                    // position), unbounded, so int64 arithmetic on it is
                    // Python's even for a fixed-width NumPy value.
                    crate::helpers::dtype::to_int64(&mut self.builder, &visited_args[0])
                }
            }
            "float" => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_float(0.0)
                } else {
                    reject_complex_conversion("float", &visited_args[0]);
                    self.builder.ir_float_cast(&visited_args[0])
                }
            }
            "bool" => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_bool(false)
                } else {
                    self.builder.ir_bool_cast(&visited_args[0])
                }
            }
            "complex" => {
                // complex() / complex(re) / complex(re, im) — Python's built-in.
                let real_v = visited_args
                    .first()
                    .cloned()
                    .unwrap_or_else(|| self.builder.ir_constant_float(0.0));
                let imag_v = visited_args
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| self.builder.ir_constant_float(0.0));
                // Promote real and imag to Float (allow int / float / complex inputs).
                let real_f = match &real_v {
                    Value::Float(_) => real_v.clone(),
                    Value::Complex { real, .. } => Value::Float(real.clone()),
                    _ => self.builder.ir_float_cast(&real_v),
                };
                let imag_f = match &imag_v {
                    Value::Float(_) => imag_v.clone(),
                    Value::Complex { imag, .. } => Value::Float(imag.clone()),
                    _ => self.builder.ir_float_cast(&imag_v),
                };
                let r = match real_f {
                    Value::Float(s) => s,
                    _ => unreachable!(),
                };
                let i = match imag_f {
                    Value::Float(s) => s,
                    _ => unreachable!(),
                };
                Value::Complex { real: r, imag: i }
            }
            "abs" => match visited_args_orig.first() {
                Some(x) => builtin_abs(&mut self.builder, x),
                None => Value::None,
            },
            "print" => {
                if !visited_args.is_empty() {
                    // For composites, print each element
                    let arg = &visited_args[0];
                    let flat = crate::helpers::composite::flatten_composite(arg);
                    if let Some(first) = flat.first() {
                        let fmt = self.builder.ir_constant_str(String::new());
                        self.builder.ir_print(first, &fmt)
                    } else {
                        Value::None
                    }
                } else {
                    Value::None
                }
            }
            "enumerate" => {
                // enumerate(iterable) → list of (index, element) tuples
                if let Some(iter_val) = visited_args_orig.first() {
                    crate::ops::static_ndarray_ops::builtin_enumerate(&mut self.builder, iter_val)
                } else {
                    Value::None
                }
            }
            "sum" => {
                if let Some(iter_val) = visited_args.first() {
                    // Python sum() iterates over the first level
                    let mut result = if let Value::DynamicNDArray(data) = iter_val {
                        assert_eq!(data.envelope.dims.len(), 1, "sum: dynamic iterable must have rank one");
                        crate::ops::dyn_ndarray::aggregation::dyn_aggregate_all(
                            &mut self.builder, data, crate::ops::dyn_ndarray::DynAggKind::Sum)
                    } else if let Value::List(data) | Value::Tuple(data) = iter_val {
                        // Check if elements are composites (2D+ array)
                        if !data.values.is_empty()
                            && matches!(&data.values[0], Value::List(_) | Value::Tuple(_))
                        {
                            // Sum over first axis (element-wise add of rows)
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = crate::ops::static_ndarray_ops::elementwise_binary(
                                    &mut self.builder,
                                    "add",
                                    &acc,
                                    row,
                                );
                            }
                            acc
                        } else {
                            crate::helpers::ndarray::builtin_reduce(
                                &mut self.builder,
                                "sum",
                                iter_val,
                            )
                        }
                    } else {
                        crate::helpers::ndarray::builtin_reduce(&mut self.builder, "sum", iter_val)
                    };
                    // sum(iterable, start) — add start value
                    if let Some(start) = visited_args_orig.get(1).or_else(|| ctx.kwargs_orig.get("start")) {
                        if matches!(start, Value::StaticArray { .. }) {
                            result = crate::helpers::static_array::base::to_static_array(&mut self.builder, &result).unwrap_or(result);
                        }
                        result = crate::helpers::value_ops::apply_binary_op(
                            &mut self.builder,
                            "add",
                            start,
                            &result,
                        );
                    }
                    result
                } else {
                    Value::None
                }
            }
            "any" | "all" => {
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, Python's any()/all() tries to evaluate truth of rows
                    // which is ambiguous for multi-element arrays
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty()
                            && matches!(&data.values[0], Value::List(_) | Value::Tuple(_))
                        {
                            panic!("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                        }
                    }
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, member, iter_val)
                } else {
                    Value::None
                }
            }
            "min" | "max" => {
                // Python's 2+-arg form `min(a, b, ...)` reduces over its
                // positional args (NOT over a single iterable). Lower via a
                // chained scalar select so each pair compiles to
                // `select(lt(a, b), a, b)` / `select(lt(a, b), b, a)`.
                // Without this branch, the prior code took
                // `visited_args.first()` only and silently dropped extra
                // args — turning `min(high, shape[1])` into a no-op clamp
                // that lets out-of-range witnesses through the slice's
                // bound discharge.
                if visited_args.len() >= 2 {
                    return Some(builtin_min_max_of_arguments(
                        &mut self.builder,
                        member,
                        &ctx.args_orig,
                    ));
                }
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, reduce over the first axis
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty()
                            && matches!(&data.values[0], Value::List(_) | Value::Tuple(_))
                        {
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = crate::ops::static_ndarray_ops::elementwise_minmax(
                                    &mut self.builder,
                                    &acc,
                                    row,
                                    member == "max",
                                );
                            }
                            return Some(acc);
                        }
                    }
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, member, iter_val)
                } else {
                    Value::None
                }
            }
            "pow" => {
                if visited_args.len() >= 3 {
                    // pow(base, exp, mod) — modular exponentiation
                    let base_exp = crate::helpers::value_ops::apply_scalar_binary_op(
                        &mut self.builder,
                        "pow",
                        &visited_args[0],
                        &visited_args[1],
                    );
                    self.builder.ir_mod_i(&base_exp, &visited_args[2])
                } else if visited_args.len() >= 2 {
                    crate::helpers::value_ops::apply_scalar_binary_op(
                        &mut self.builder,
                        "pow",
                        &visited_args[0],
                        &visited_args[1],
                    )
                } else {
                    Value::None
                }
            }
            "list" | "tuple" => {
                let values = match visited_args.first() {
                    None => Vec::new(),
                    Some(_) => self.iterated_values(&ctx.args_orig[0], member),
                };
                let data = CompositeData {
                    elements_type: values.iter().map(|v| v.zinnia_type()).collect(),
                    values,
                    // `list(a)` is a new (shallow) list, not `a` itself.
                    value_id: ValueId::next(),
                };
                if member == "list" {
                    Value::List(data)
                } else {
                    Value::Tuple(data)
                }
            }
            _ => return None,
        };
        Some(v)
    }
}

/// Python's `abs(x)`: the absolute value of a number (the magnitude of a
/// complex number), and elementwise for an ndarray (`ndarray.__abs__`,
/// which keeps the dtype; the magnitude array of a complex one is float).
/// A float is a fixed-point cell: taking it with the integer `abs` made the
/// result an integer holding 2^32 times the value. A Python list or tuple
/// has no `abs` (`TypeError`).
fn builtin_abs(builder: &mut crate::builder::IRBuilder, x: &Value) -> Value {
    use crate::types::NumberType;
    match x {
        Value::Integer(_) | Value::Boolean(_) => builder.ir_abs_i(x),
        Value::Float(_) => builder.ir_abs_f(x),
        // |a + bi| = sqrt(a^2 + b^2)
        Value::Complex { real, imag } => {
            let r = Value::Float(real.clone());
            let i = Value::Float(imag.clone());
            let rr = builder.ir_mul_f(&r, &r);
            let ii = builder.ir_mul_f(&i, &i);
            let sum = builder.ir_add_f(&rr, &ii);
            builder.ir_sqrt_f(&sum)
        }
        Value::StaticArray { dtype: NumberType::Complex, .. } => {
            crate::helpers::static_array::complex::np_abs_complex_static_array(builder, x)
        }
        Value::StaticArray { dtype, shape, .. } => {
            let cells = crate::helpers::static_array::elementwise::payload_cells(builder, x);
            let magnitudes = cells.iter().map(|cell| builtin_abs(builder, cell)).collect();
            crate::helpers::static_array::base::build_static_array_from_flat(
                builder,
                magnitudes,
                shape.clone(),
                *dtype,
            )
        }
        Value::DynamicNDArray(data) if data.dtype != NumberType::Complex => {
            crate::ops::dyn_ndarray::binary::dyn_unary_op(builder, data, data.dtype, builtin_abs)
        }
        Value::DynamicNDArray(_) => {
            panic!("abs() of a complex array with a runtime shape is not supported")
        }
        other => panic!("TypeError: bad operand type for abs(): '{}'", other.zinnia_type()),
    }
}

/// Python's `min(a, b, ...)` / `max(a, b, ...)` over real scalars: the first
/// argument, replaced by each later one that is strictly smaller (`min`) or
/// strictly greater (`max`). Integers and booleans compare and select as
/// integers. When either side of a step is a float, both sides are compared
/// and selected as floats: a float is a fixed-point cell, and selecting it
/// with the integer select made the result an integer holding 2^32 times the
/// value (`max(1.0, x) + 1e-9` came out as about `x * 2^32`).
fn builtin_min_max_of_arguments(
    builder: &mut crate::builder::IRBuilder,
    function: &str,
    arguments: &[Value],
) -> Value {
    for argument in arguments {
        match argument {
            Value::Integer(_) | Value::Boolean(_) | Value::Float(_) => {}
            Value::Complex { .. } => panic!(
                "TypeError: {function}(a, b, ...) cannot order complex numbers"
            ),
            Value::StaticArray { .. } | Value::DynamicNDArray(_) => panic!(
                "{function}(a, b, ...) of arrays is not supported: comparing two arrays has no \
                 single truth value in NumPy; use np.{function}imum(a, b) for the elementwise \
                 {function}"
            ),
            other => panic!(
                "{function}(a, b, ...) is supported for numbers only, got {} (comparing \
                 sequences lexicographically is not supported)",
                other.zinnia_type()
            ),
        }
    }
    let mut acc = arguments[0].clone();
    for next in &arguments[1..] {
        let (smaller, larger) = if function == "min" { (next, &acc) } else { (&acc, next) };
        let replace =
            crate::helpers::value_ops::apply_scalar_binary_op(builder, "lt", smaller, larger);
        acc = if matches!(acc, Value::Float(_)) || matches!(next, Value::Float(_)) {
            let next_f = crate::helpers::value_ops::ensure_float(builder, next);
            let acc_f = crate::helpers::value_ops::ensure_float(builder, &acc);
            builder.ir_select_f(&replace, &next_f, &acc_f)
        } else {
            builder.ir_select_i(&replace, next, &acc)
        };
    }
    acc
}

/// Python's `int(z)` / `float(z)` raise `TypeError` for a complex `z`.
fn reject_complex_conversion(function: &str, value: &Value) {
    if matches!(value, Value::Complex { .. }) {
        panic!("TypeError: {function}() argument must be a real number, not 'complex' (use .real)");
    }
}
