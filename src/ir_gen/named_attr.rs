use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, Value, ZinniaType};

use super::IRGenerator;

impl IRGenerator {
    pub(crate) fn visit_named_attr(&mut self, n: &ASTNamedAttribute) -> Value {
        // Handle starred args: foo(1, *args, 4) → flatten starred into arg list
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    visited_args.extend(data.values);
                } else {
                    visited_args.push(inner);
                }
            } else {
                visited_args.push(self.visit(a));
            }
        }
        let _visited_kwargs: HashMap<String, Value> = n
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), self.visit(v)))
            .collect();

        let target = n.target.as_deref();
        let member = n.member.as_str();

        match (target, member) {
            // ── Built-in functions (no target) ─────────────────────────
            (None, "range") => crate::ops::static_ndarray_ops::builtin_range(&mut self.builder, &visited_args),
            (None, "len") => crate::ops::static_ndarray_ops::builtin_len(&mut self.builder, &visited_args),
            (None, "int") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_int(0)
                } else {
                    self.builder.ir_int_cast(&visited_args[0])
                }
            }
            (None, "float") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_float(0.0)
                } else {
                    self.builder.ir_float_cast(&visited_args[0])
                }
            }
            (None, "bool") => {
                if visited_args.is_empty() {
                    self.builder.ir_constant_bool(false)
                } else {
                    self.builder.ir_bool_cast(&visited_args[0])
                }
            }
            (None, "abs") => {
                if !visited_args.is_empty() {
                    match &visited_args[0] {
                        Value::List(data) | Value::Tuple(data) => {
                            let results: Vec<Value> = data.values.iter()
                                .map(|v| self.builder.ir_abs_i(v))
                                .collect();
                            let types = results.iter().map(|v| v.zinnia_type()).collect();
                            Value::List(CompositeData { elements_type: types, values: results })
                        }
                        v => self.builder.ir_abs_i(v),
                    }
                } else {
                    Value::None
                }
            }
            (None, "print") => {
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
            (None, "enumerate") => {
                // enumerate(iterable) → list of (index, element) tuples
                if let Some(iter_val) = visited_args.first() {
                    crate::ops::static_ndarray_ops::builtin_enumerate(&mut self.builder, iter_val)
                } else {
                    Value::None
                }
            }
            (None, "sum") => {
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
            (None, "any") | (None, "all") => {
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
            (None, "min") | (None, "max") => {
                if let Some(iter_val) = visited_args.first() {
                    // For 2D+ arrays, reduce over the first axis
                    if let Value::List(data) | Value::Tuple(data) = iter_val {
                        if !data.values.is_empty() && matches!(&data.values[0], Value::List(_) | Value::Tuple(_)) {
                            let mut acc = data.values[0].clone();
                            for row in &data.values[1..] {
                                acc = crate::ops::static_ndarray_ops::elementwise_minmax(&mut self.builder, &acc, row, member == "max");
                            }
                            return acc;
                        }
                    }
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, member, iter_val)
                } else {
                    Value::None
                }
            }
            (None, "pow") => {
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
            (None, "list") => {
                // list(iterable) — convert to list, or empty list
                if visited_args.is_empty() {
                    Value::List(CompositeData { elements_type: vec![], values: vec![] })
                } else {
                    match &visited_args[0] {
                        Value::List(_) => visited_args[0].clone(),
                        Value::Tuple(data) => {
                            Value::List(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }
            (None, "tuple") => {
                // tuple(iterable) — convert to tuple, or empty tuple
                if visited_args.is_empty() {
                    Value::Tuple(CompositeData { elements_type: vec![], values: vec![] })
                } else {
                    match &visited_args[0] {
                        Value::Tuple(_) => visited_args[0].clone(),
                        Value::List(data) => {
                            Value::Tuple(CompositeData {
                                elements_type: data.elements_type.clone(),
                                values: data.values.clone(),
                            })
                        }
                        _ => visited_args[0].clone(),
                    }
                }
            }

            // ── np.* (numpy-like operations) ───────────────────────────
            (Some("np"), "asarray") => {
                // np.asarray(x) — pass through if already a composite
                if !visited_args.is_empty() {
                    let val = &visited_args[0];
                    // Validate that all sub-elements have consistent shapes
                    if let Value::List(data) | Value::Tuple(data) = val {
                        if data.values.len() > 1 {
                            let first_is_composite = matches!(&data.values[0], Value::List(_) | Value::Tuple(_));
                            for v in &data.values[1..] {
                                let is_composite = matches!(v, Value::List(_) | Value::Tuple(_));
                                if is_composite != first_is_composite {
                                    panic!("To convert to NDArray, all sub-lists should be of the same shape");
                                }
                            }
                        }
                    }
                    // Handle dtype kwarg for type casting
                    if let Some(dtype) = _visited_kwargs.get("dtype") {
                        let to_float = matches!(dtype, Value::Class(ZinniaType::Float));
                        self.cast_composite(val, to_float)
                    } else {
                        val.clone()
                    }
                } else {
                    Value::None
                }
            }
            (Some("np"), "zeros") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, &visited_args, &_visited_kwargs, 0),
            (Some("np"), "ones") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, &visited_args, &_visited_kwargs, 1),
            (Some("np"), "identity") => crate::ops::static_ndarray_ops::np_identity(&mut self.builder, &visited_args),
            (Some("np"), "arange") => crate::ops::static_ndarray_ops::np_arange(&mut self.builder, &visited_args),
            (Some("np"), "linspace") => crate::ops::static_ndarray_ops::np_linspace(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "allclose") => crate::ops::static_ndarray_ops::np_allclose(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "concatenate") => crate::ops::static_ndarray_ops::np_concatenate(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "stack") => crate::ops::static_ndarray_ops::np_stack(&mut self.builder, &visited_args, &_visited_kwargs),

            // ── DynamicNDArray class methods ────────────────────────────
            (Some("DynamicNDArray"), "zeros") | (Some("zinnia"), "zeros") => {
                crate::ops::dyn_ndarray::metadata::dyn_zeros(&mut self.builder, &visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "ones") | (Some("zinnia"), "ones") => {
                crate::ops::dyn_ndarray::metadata::dyn_ones(&mut self.builder, &visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "eye") | (Some("zinnia"), "eye") => {
                crate::ops::dyn_ndarray::constructors::dyn_eye(&mut self.builder, &visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "concatenate") | (Some("zinnia"), "concatenate") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(&mut self.builder, &visited_args, &_visited_kwargs)
            }
            (Some("DynamicNDArray"), "stack") | (Some("zinnia"), "stack") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_stack(&mut self.builder, &visited_args, &_visited_kwargs)
            }

            // ── List methods (target is a variable name) ───────────────
            (Some(var), "append") if self.ctx.exists(var) => {
                self.list_method_append(var, &visited_args)
            }
            (Some(var), "extend") if self.ctx.exists(var) => {
                self.list_method_extend(var, &visited_args)
            }
            (Some(var), "pop") if self.ctx.exists(var) => {
                self.list_method_pop(var, &visited_args)
            }
            (Some(var), "remove") if self.ctx.exists(var) => {
                self.list_method_remove(var, &visited_args)
            }
            (Some(var), "clear") if self.ctx.exists(var) => {
                self.list_method_clear(var)
            }
            (Some(var), "copy") if self.ctx.exists(var) => {
                self.ctx.get(var).unwrap_or(Value::None)
            }
            (Some(var), "reverse") if self.ctx.exists(var) => {
                self.list_method_reverse(var)
            }
            (Some(var), "count") if self.ctx.exists(var) => {
                self.list_method_count(var, &visited_args)
            }
            (Some(var), "index") if self.ctx.exists(var) => {
                self.list_method_index(var, &visited_args)
            }
            (Some(var), "insert") if self.ctx.exists(var) => {
                self.list_method_insert(var, &visited_args)
            }

            // ── DynamicNDArray method dispatch ────────────────────────
            (Some(var), method) if self.ctx.exists(var) && matches!(self.ctx.get(var), Some(Value::DynamicNDArray(_))) => {
                let val = self.ctx.get(var).unwrap();
                self.dispatch_dyn_ndarray_method(val, method, &visited_args, &_visited_kwargs)
            }

            // ── Method calls on expr attributes ────────────────────────
            (Some(var), "sum") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"sum", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "sum", &val)
                }
            }
            (Some(var), "any") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"any", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "any", &val)
                }
            }
            (Some(var), "all") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"all", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "all", &val)
                }
            }
            (Some(var), "transpose") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // Check for axes keyword argument
                let args = if let Some(axes_val) = _visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &args)
            }
            (Some(var), "T") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &[])
            }
            (Some(var), "tolist") if self.ctx.exists(var) => {
                self.ctx.get(var).unwrap_or(Value::None)
            }
            (Some(var), "astype") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // Determine target type from the argument (int or float class)
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&val, target_float)
            }
            (Some(var), "argmax") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::ndarray_argmax_argmin_with_axis(&mut self.builder,&val, ax, true)
                } else {
                    crate::helpers::ndarray::ndarray_argmax_argmin(&mut self.builder, &val, &visited_args, true)
                }
            }
            (Some(var), "argmin") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::ndarray_argmax_argmin_with_axis(&mut self.builder,&val, ax, false)
                } else {
                    crate::helpers::ndarray::ndarray_argmax_argmin(&mut self.builder, &val, &visited_args, false)
                }
            }

            // ── NDArray property access ──────────────────────────────
            (Some(var), "shape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                let shape_vals: Vec<Value> = shape.iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = vec![ZinniaType::Integer; shape_vals.len()];
                Value::Tuple(CompositeData { elements_type: types, values: shape_vals })
            }
            (Some(var), "dtype") if self.ctx.exists(var) => {
                // Infer dtype from element types
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float {
                    Value::Class(ZinniaType::Float)
                } else {
                    Value::Class(ZinniaType::Integer)
                }
            }
            (Some(var), "min") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"min", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "min", &val)
                }
            }
            (Some(var), "max") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"max", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "max", &val)
                }
            }
            (Some(var), "prod") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis = _visited_kwargs.get("axis").or_else(|| visited_args.first()).and_then(|v| v.int_val());
                if let Some(ax) = axis {
                    crate::ops::static_ndarray_ops::reduce_with_axis(&mut self.builder,"prod", &val, ax)
                } else {
                    crate::helpers::ndarray::builtin_reduce(&mut self.builder, "prod", &val)
                }
            }

            // ── NDArray ndim, size, flatten, flat, reshape, moveaxis, repeat, filter ─
            (Some(var), "ndim") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            (Some(var), "size") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            (Some(var), "flatten") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "flat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "reshape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_reshape(&mut self.builder,&val, &visited_args)
            }
            (Some(var), "moveaxis") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_moveaxis(&mut self.builder,&val, &visited_args)
            }
            (Some(var), "repeat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_repeat(&mut self.builder,&val, &visited_args, &_visited_kwargs)
            }
            (Some(var), "filter") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_filter(&mut self.builder,&val, &visited_args)
            }

            // ── Chip calls (no target or target not a known variable) ─
            (None, name) if self.registered_chips.contains_key(name) => {
                self.visit_chip_call(name, &visited_args, &_visited_kwargs)
            }

            // ── External function calls ─────────────────────────────────
            (None, name) if self.registered_externals.contains_key(name) => {
                self.visit_external_call(name, &visited_args)
            }

            // ── Fallback ───────────────────────────────────────────────
            _ => {
                panic!(
                    "Named attribute `{}.{}` not yet implemented in Rust IR generator",
                    target.unwrap_or(""),
                    member
                )
            }
        }
    }

    pub(crate) fn visit_expr_attr(&mut self, n: &ASTExprAttribute) -> Value {
        let target = self.visit(&n.target);
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    visited_args.extend(data.values);
                } else {
                    visited_args.push(inner);
                }
            } else {
                visited_args.push(self.visit(a));
            }
        }
        let visited_kwargs: HashMap<String, Value> = n
            .kwargs
            .iter()
            .map(|(k, v)| (k.clone(), self.visit(v)))
            .collect();

        // DynamicNDArray dispatch — route to dedicated handler
        if matches!(target, Value::DynamicNDArray(_)) {
            return self.dispatch_dyn_ndarray_method(
                target, n.member.as_str(), &visited_args, &visited_kwargs,
            );
        }

        match n.member.as_str() {
            "sum" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "sum", &target),
            "any" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "any", &target),
            "all" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "all", &target),
            "transpose" => {
                let args = if let Some(axes_val) = visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &target, &args)
            }
            "T" => crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &target, &[]),
            "tolist" => target,
            "astype" => {
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&target, target_float)
            }
            "argmax" => crate::helpers::ndarray::ndarray_argmax_argmin(&mut self.builder, &target, &visited_args, true),
            "argmin" => crate::helpers::ndarray::ndarray_argmax_argmin(&mut self.builder, &target, &visited_args, false),
            "prod" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "prod", &target),
            "min" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "min", &target),
            "max" => crate::helpers::ndarray::builtin_reduce(&mut self.builder, "max", &target),
            "ndim" => {
                let shape = crate::helpers::composite::get_composite_shape(&target);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            "size" => {
                let shape = crate::helpers::composite::get_composite_shape(&target);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            "flatten" | "flat" => {
                let flat = crate::helpers::composite::flatten_composite(&target);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            "reshape" => crate::ops::static_ndarray_ops::ndarray_reshape(&mut self.builder,&target, &visited_args),
            "moveaxis" => crate::ops::static_ndarray_ops::ndarray_moveaxis(&mut self.builder,&target, &visited_args),
            "repeat" => crate::ops::static_ndarray_ops::ndarray_repeat(&mut self.builder,&target, &visited_args, &visited_kwargs),
            "filter" => crate::ops::static_ndarray_ops::ndarray_filter(&mut self.builder,&target, &visited_args),
            "shape" => {
                let shape = crate::helpers::composite::get_composite_shape(&target);
                let shape_vals: Vec<Value> = shape.iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = vec![ZinniaType::Integer; shape_vals.len()];
                Value::Tuple(CompositeData { elements_type: types, values: shape_vals })
            }
            "dtype" => {
                let flat = crate::helpers::composite::flatten_composite(&target);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float { Value::Class(ZinniaType::Float) } else { Value::Class(ZinniaType::Integer) }
            }
            "append" | "extend" | "pop" | "remove" | "clear" |
            "copy" | "reverse" | "count" | "index" => {
                // These should be handled via visit_named_attr with a target variable name
                panic!("Expr attribute `.{}` on non-variable target not supported", n.member)
            }
            _ => {
                panic!(
                    "Expr attribute `.{}` not yet implemented in Rust IR generator",
                    n.member
                )
            }
        }
    }
}
