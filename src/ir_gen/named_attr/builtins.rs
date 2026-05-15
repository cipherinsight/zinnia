//! Python-builtin dispatch arms (no target on the named-attribute call):
//! `range`, `len`, `int`, `float`, `bool`, `complex`, `abs`, `print`,
//! `enumerate`, `sum`, `any`, `all`, `min`, `max`, `pow`, `list`, `tuple`.
//!
//! Returns `None` when the (target, member) pair is not a recognized builtin,
//! so `visit_named_attr` can fall through to the next category.

use crate::types::{CompositeData, Value, ValueId};

use super::DispatchCtx;
use super::super::IRGenerator;

impl IRGenerator {
    pub(crate) fn try_builtin(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        if ctx.target.is_some() {
            return None;
        }
        let visited_args = ctx.args;
        let visited_args_orig = ctx.args_orig;
        let member = ctx.member;
        let v = match member {
            "range" => crate::ops::static_ndarray_ops::builtin_range(&mut self.builder, visited_args),
            "len" => crate::ops::static_ndarray_ops::builtin_len(&mut self.builder, visited_args),
            "int" => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_int(0)
                } else {
                    self.builder.ir_int_cast(&visited_args[0])
                }
            }
            "float" => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_float(0.0)
                } else {
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
                let real_v = visited_args.first().cloned().unwrap_or_else(|| self.builder.ir_constant_float(0.0));
                let imag_v = visited_args.get(1).cloned().unwrap_or_else(|| self.builder.ir_constant_float(0.0));
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
                let r = match real_f { Value::Float(s) => s, _ => unreachable!() };
                let i = match imag_f { Value::Float(s) => s, _ => unreachable!() };
                Value::Complex { real: r, imag: i }
            }
            "abs" => {
                if !visited_args.is_empty() {
                    let arg_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                    // P5a: Complex StaticArray → fresh Float StaticArray.
                    if let Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } = &arg_orig {
                        return Some(crate::helpers::static_array_complex::np_abs_complex_static_array(&mut self.builder, &arg_orig));
                    }
                    match &visited_args[0] {
                        Value::List(data) | Value::Tuple(data) => {
                            let results: Vec<Value> = data.values.iter()
                                .map(|v| self.builder.ir_abs_i(v))
                                .collect();
                            let types = results.iter().map(|v| v.zinnia_type()).collect();
                            Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() })
                        }
                        // |a + bi| = sqrt(a^2 + b^2)
                        Value::Complex { real, imag } => {
                            let r = Value::Float(real.clone());
                            let i = Value::Float(imag.clone());
                            let rr = self.builder.ir_mul_f(&r, &r);
                            let ii = self.builder.ir_mul_f(&i, &i);
                            let sum = self.builder.ir_add_f(&rr, &ii);
                            self.builder.ir_sqrt_f(&sum)
                        }
                        v => self.builder.ir_abs_i(v),
                    }
                } else {
                    Value::None
                }
            }
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
                if let Some(iter_val) = visited_args.first() {
                    crate::ops::static_ndarray_ops::builtin_enumerate(&mut self.builder, iter_val)
                } else {
                    Value::None
                }
            }
            "sum" => {
                if let Some(iter_val) = visited_args.first() {
                    // Python sum() iterates over the first level
                    let mut result = if let Value::List(data) | Value::Tuple(data) = iter_val {
                        // Check if elements are composites (2D+ array)
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            // Sum over first axis (element-wise add of rows)
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = crate::ops::static_ndarray_ops::elementwise_binary(&mut self.builder, "add", &acc, row);
                            }
                            acc
                        } else {
                            crate::helpers::ndarray::builtin_reduce(&mut self.builder, "sum", iter_val)
                        }
                    } else {
                        crate::helpers::ndarray::builtin_reduce(&mut self.builder, "sum", iter_val)
                    };
                    // sum(iterable, start) — add start value
                    if let Some(start) = visited_args.get(1) {
                        result = crate::helpers::value_ops::apply_binary_op(&mut self.builder, "add", start, &result);
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
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            panic!("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                        }
                    }
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, member, iter_val)
                } else {
                    Value::None
                }
            }
            "min" | "max" => {
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, reduce over the first axis
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = crate::ops::static_ndarray_ops::elementwise_minmax(&mut self.builder, &acc, row, member == "max");
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
                    let base_exp = crate::helpers::value_ops::apply_scalar_binary_op(&mut self.builder, "pow", &visited_args[0], &visited_args[1]);
                    self.builder.ir_mod_i(&base_exp, &visited_args[2])
                } else if visited_args.len() >= 2 {
                    crate::helpers::value_ops::apply_scalar_binary_op(&mut self.builder, "pow", &visited_args[0], &visited_args[1])
                } else {
                    Value::None
                }
            }
            "list" => {
                // list(iterable) — convert to list, or empty list
                if visited_args.is_empty() {
                    Value::List(CompositeData { elements_type: vec![], values: vec![], value_id: ValueId::next() })
                } else {
                    match &visited_args[0] {
                        Value::List(_) => visited_args[0].clone(),
                        Value::Tuple(data) => {
                            Value::List(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),

                                value_id: ValueId::next(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }
            "tuple" => {
                // tuple(iterable) — convert to tuple, or empty tuple
                if visited_args.is_empty() {
                    Value::Tuple(CompositeData { elements_type: vec![], values: vec![], value_id: ValueId::next() })
                } else {
                    match &visited_args[0] {
                        Value::Tuple(_) => visited_args[0].clone(),
                        Value::List(data) => {
                            Value::Tuple(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),

                                value_id: ValueId::next(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }
            _ => return None,
        };
        Some(v)
    }
}
