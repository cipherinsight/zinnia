use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, ScalarValue, Value, ZinniaType};

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

        // P1 segarr boundary: legacy ops still pattern-match `Value::List` /
        // `Value::Tuple` for numeric arrays. Materialise any segment-backed
        // `StaticArray` (top-level or nested inside a List/Tuple) into the
        // nested-List representation. Constructors emit StaticArray; that
        // happens *inside* the per-op handler, so the conversion here only
        // affects values that are flowing in as op inputs.
        //
        // Scope-keeping note: this is a coarse top-of-dispatcher shim. Once
        // ops migrate to the segment representation natively (P2 onward),
        // each migrated op opts out by pattern-matching `Value::StaticArray`
        // before this conversion fires.
        //
        // P4b: keep the un-converted args around so the migrated reduction
        // entry points can route through `static_array_reductions` directly
        // on the StaticArray representation.
        let visited_args_orig: Vec<Value> = visited_args.clone();
        let visited_kwargs_orig: HashMap<String, Value> = _visited_kwargs.clone();
        let visited_args: Vec<Value> = visited_args
            .iter()
            .map(|v| crate::helpers::static_array::deep_to_value_list(&mut self.builder, v))
            .collect();
        let _visited_kwargs: HashMap<String, Value> = _visited_kwargs
            .into_iter()
            .map(|(k, v)| (k, crate::helpers::static_array::deep_to_value_list(&mut self.builder, &v)))
            .collect();

        // Helper: check if the first arg is a list/tuple containing DynamicNDArrays.
        fn has_dynamic_array_in_list(args: &[Value]) -> bool {
            match args.first() {
                Some(Value::List(cd)) | Some(Value::Tuple(cd)) => {
                    cd.values.iter().any(|v| matches!(v, Value::DynamicNDArray(_)))
                }
                _ => false,
            }
        }
        // P4c: helper to detect StaticArray inputs in a list/tuple — used to
        // decide whether to route concatenate/stack/etc. through the native
        // path before falling back to the legacy nested-List handlers.
        fn list_contains_static_array(arg: &Value) -> bool {
            match arg {
                Value::List(cd) | Value::Tuple(cd) => {
                    cd.values.iter().any(|v| matches!(v, Value::StaticArray { .. }))
                }
                _ => false,
            }
        }

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
            (None, "complex") => {
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
            (None, "abs") => {
                if !visited_args.is_empty() {
                    let arg_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                    // P5a: Complex StaticArray → fresh Float StaticArray.
                    if let Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } = &arg_orig {
                        return crate::helpers::static_array_complex::np_abs_complex_static_array(&mut self.builder, &arg_orig);
                    }
                    match &visited_args[0] {
                        Value::List(data) | Value::Tuple(data) => {
                            let results: Vec<Value> = data.values.iter()
                                .map(|v| self.builder.ir_abs_i(v))
                                .collect();
                            let types = results.iter().map(|v| v.zinnia_type()).collect();
                            Value::List(CompositeData { elements_type: types, values: results })
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
            // numpy dtype aliases — collapse width into ZinniaType (Float / Integer / Boolean)
            (Some("np"), "float16" | "float32" | "float64" | "double" | "single" | "half" | "longdouble") => Value::Class(ZinniaType::Float),
            (Some("np"), "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
                       | "intp" | "uintp" | "long" | "byte" | "short" | "intc" | "uintc" | "ubyte"
                       | "ushort" | "longlong" | "ulonglong") => Value::Class(ZinniaType::Integer),
            (Some("np"), "bool_") => Value::Class(ZinniaType::Boolean),
            (Some("np"), "complex64" | "complex128" | "complex256" | "csingle" | "cdouble" | "clongdouble") => Value::Class(ZinniaType::Complex),
            // Complex helpers
            (Some("np"), "conj" | "conjugate") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { real, imag } => {
                        // (a + bi) -> (a - bi)
                        let zero = self.builder.ir_constant_float(0.0);
                        let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag));
                        let ni = match neg_imag {
                            Value::Float(s) => s,
                            _ => unreachable!(),
                        };
                        Value::Complex { real, imag: ni }
                    }
                    // P5a: Complex StaticArray → fresh imag segment with negated values; share real segment.
                    Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } => {
                        crate::helpers::static_array_complex::np_conj_static_array(&mut self.builder, &v)
                    }
                    // For real inputs, conj is the identity.
                    other => other,
                }
            }
            (Some("np"), "real") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { real, .. } => Value::Float(real),
                    Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } => {
                        crate::helpers::static_array_complex::np_real_static_array(&mut self.builder, &v)
                    }
                    // For real inputs, real is the identity.
                    other => other,
                }
            }
            (Some("np"), "imag") => {
                let v = visited_args_orig.first().cloned().unwrap_or(Value::None);
                match v {
                    Value::Complex { imag, .. } => Value::Float(imag),
                    Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } => {
                        crate::helpers::static_array_complex::np_imag_static_array(&mut self.builder, &v)
                    }
                    // For real inputs, imag is zero of same shape.
                    Value::StaticArray { shape, .. } => {
                        let total: usize = shape.iter().product();
                        let zero = self.builder.ir_constant_float(0.0);
                        crate::helpers::static_array::build_static_array_from_flat(
                            &mut self.builder,
                            vec![zero; total],
                            shape,
                            crate::types::NumberType::Float,
                        )
                    }
                    Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => self.builder.ir_constant_float(0.0),
                    other => other,
                }
            }
            // numpy / math mathematical constants
            (Some("np"), "pi") | (Some("math"), "pi") => self.builder.ir_constant_float(std::f64::consts::PI),
            (Some("np"), "e") | (Some("math"), "e") => self.builder.ir_constant_float(std::f64::consts::E),
            (Some("np"), "asarray" | "array") => {
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
                    let cast_val = if let Some(dtype) = _visited_kwargs.get("dtype") {
                        let to_float = matches!(dtype, Value::Class(ZinniaType::Float));
                        self.cast_composite(val, to_float)
                    } else {
                        val.clone()
                    };
                    // P1 segarr-foundation: numeric Python list/tuple inputs
                    // become segment-backed `Value::StaticArray`. Falls back
                    // to passing through for non-numeric / heterogeneous data.
                    if let Some(sa) = crate::helpers::static_array::to_static_array(&mut self.builder, &cast_val) {
                        sa
                    } else {
                        cast_val
                    }
                } else {
                    Value::None
                }
            }
            (Some("np"), "zeros") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, &visited_args, &_visited_kwargs, 0),
            (Some("np"), "ones") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, &visited_args, &_visited_kwargs, 1),
            (Some("np"), "empty" | "ndarray") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, &visited_args, &_visited_kwargs, 0),
            (Some("np"), "zeros_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, &visited_args, &_visited_kwargs, 0),
            (Some("np"), "ones_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, &visited_args, &_visited_kwargs, 1),
            (Some("np"), "empty_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, &visited_args, &_visited_kwargs, 0),
            (Some("np"), "identity" | "eye") => crate::ops::static_ndarray_ops::np_identity(&mut self.builder, &visited_args),
            // numpy reduction aliases — forward to the existing reduce/argmax_argmin path used by x.sum() / x.argmin()
            (Some("np"), method @ ("sum" | "prod" | "min" | "max" | "any" | "all" | "mean")) => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_reduce(
                    &mut self.builder,
                    method,
                    &val_orig,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return out;
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.get(1));
                crate::helpers::array_ops::reduce(&mut self.builder, method, &val, axis_arg)
            }
            (Some("np"), method @ ("argmax" | "argmin")) => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.get(1));
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &val_orig,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return out;
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.get(1));
                crate::helpers::array_ops::argmax_argmin(&mut self.builder, &val, axis_arg, method == "argmax")
            }
            (Some("np"), "dot" | "matmul") => {
                let lhs = visited_args.first().cloned().unwrap_or(Value::None);
                let rhs = visited_args.get(1).cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::matmul(&mut self.builder, &lhs, &rhs)
            }
            (Some("np"), "square") => crate::ops::static_ndarray_ops::np_square(&mut self.builder, &visited_args),
            (Some("np"), "diff") => crate::ops::static_ndarray_ops::np_diff(&mut self.builder, &visited_args),
            (Some("np"), "outer") => crate::ops::static_ndarray_ops::np_outer(&mut self.builder, &visited_args),
            (Some("np"), "add_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, &visited_args, "add"),
            (Some("np"), "subtract_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, &visited_args, "sub"),
            (Some("np"), "multiply_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, &visited_args, "mul"),
            (Some("np"), "divide_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, &visited_args, "div"),
            (Some("np"), "arange") => crate::ops::static_ndarray_ops::np_arange(&mut self.builder, &visited_args),
            (Some("np"), "linspace") => crate::ops::static_ndarray_ops::np_linspace(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "allclose") => crate::ops::static_ndarray_ops::np_allclose(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "concatenate") => {
                if has_dynamic_array_in_list(&visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(&mut self.builder, &visited_args, &_visited_kwargs)
                } else {
                    // P4c: native StaticArray dispatch — uses orig args (un-converted).
                    let raw_axis = visited_kwargs_orig.get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .and_then(|v| v.int_val())
                        .unwrap_or(0);
                    if let Some(arrays_arg) = visited_args_orig.first() {
                        if list_contains_static_array(arrays_arg) {
                            if let Some(out) = crate::helpers::static_array_shape::try_apply_concatenate(
                                &mut self.builder, arrays_arg, raw_axis,
                            ) {
                                return out;
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_concatenate(&mut self.builder, &visited_args, &_visited_kwargs)
                }
            }
            (Some("np"), "stack") => {
                if has_dynamic_array_in_list(&visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_stack(&mut self.builder, &visited_args, &_visited_kwargs)
                } else {
                    // P4c: native StaticArray dispatch.
                    let raw_axis = visited_kwargs_orig.get("axis")
                        .or_else(|| visited_args_orig.get(1))
                        .and_then(|v| v.int_val())
                        .unwrap_or(0);
                    if let Some(arrays_arg) = visited_args_orig.first() {
                        if list_contains_static_array(arrays_arg) {
                            if let Some(out) = crate::helpers::static_array_shape::try_apply_stack(
                                &mut self.builder, arrays_arg, raw_axis,
                            ) {
                                return out;
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_stack(&mut self.builder, &visited_args, &_visited_kwargs)
                }
            }
            (Some("np"), "vstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_vstack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return out;
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_vstack(&mut self.builder, &visited_args)
            }
            (Some("np"), "row_stack") => crate::ops::static_ndarray_ops::np_row_stack(&mut self.builder, &visited_args),
            (Some("np"), "hstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_hstack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return out;
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_hstack(&mut self.builder, &visited_args)
            }
            (Some("np"), "dstack") => crate::ops::static_ndarray_ops::np_dstack(&mut self.builder, &visited_args),
            (Some("np"), "column_stack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_column_stack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return out;
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_column_stack(&mut self.builder, &visited_args)
            }
            (Some("np"), "swapaxes") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_swapaxes(&mut self.builder, &val, &visited_args[1..])
            }
            (Some("np"), "moveaxis") => {
                // P4c: native StaticArray dispatch via array_ops::moveaxis fast-path,
                // using the un-converted orig args.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::moveaxis(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    );
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_moveaxis(&mut self.builder, &val, &visited_args[1..])
            }
            (Some("np"), "transpose") => {
                // P4c: native StaticArray dispatch.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    ) {
                        return out;
                    }
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &visited_args[1..])
            }
            (Some("np"), "flip") => crate::ops::static_ndarray_ops::np_flip(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "flipud") => crate::ops::static_ndarray_ops::np_flipud(&mut self.builder, &visited_args),
            (Some("np"), "fliplr") => crate::ops::static_ndarray_ops::np_fliplr(&mut self.builder, &visited_args),
            (Some("np"), "rot90") => crate::ops::static_ndarray_ops::np_rot90(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "squeeze") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_kwargs_orig.get("axis").or_else(|| visited_args_orig.get(1));
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_squeeze(
                        &mut self.builder, &val_orig, axis_arg,
                    ) {
                        return out;
                    }
                }
                crate::ops::static_ndarray_ops::np_squeeze(&mut self.builder, &visited_args, &_visited_kwargs)
            }
            (Some("np"), "expand_dims") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_args_orig.get(1);
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_expand_dims(
                        &mut self.builder, &val_orig, axis_arg,
                    ) {
                        return out;
                    }
                }
                crate::ops::static_ndarray_ops::np_expand_dims(&mut self.builder, &visited_args)
            }
            (Some("np"), "broadcast_to") => crate::ops::static_ndarray_ops::np_broadcast_to(&mut self.builder, &visited_args),
            (Some("np"), "atleast_1d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, &visited_args, 1),
            (Some("np"), "atleast_2d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, &visited_args, 2),
            (Some("np"), "atleast_3d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, &visited_args, 3),
            (Some("np"), "tile") => crate::ops::static_ndarray_ops::np_tile(&mut self.builder, &visited_args),
            (Some("np"), "split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &_visited_kwargs)
                } else {
                    crate::ops::static_ndarray_ops::np_split(&mut self.builder, &visited_args, &_visited_kwargs)
                }
            }
            (Some("np"), "array_split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_array_split(&mut self.builder, &data, &visited_args[1..], &_visited_kwargs)
                } else {
                    crate::ops::static_ndarray_ops::np_array_split(&mut self.builder, &visited_args, &_visited_kwargs)
                }
            }
            (Some("np"), "hsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let ndim = data.meta.logical_shape.len();
                    let ax = if ndim == 1 { 0i64 } else { 1i64 };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert("axis".to_string(), Value::Integer(ScalarValue::new(Some(ax), None)));
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &kw)
                } else {
                    crate::ops::static_ndarray_ops::np_hsplit(&mut self.builder, &visited_args)
                }
            }
            (Some("np"), "vsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert("axis".to_string(), Value::Integer(ScalarValue::new(Some(0), None)));
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &kw)
                } else {
                    crate::ops::static_ndarray_ops::np_vsplit(&mut self.builder, &visited_args)
                }
            }
            (Some("np"), "dsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert("axis".to_string(), Value::Integer(ScalarValue::new(Some(2), None)));
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &kw)
                } else {
                    crate::ops::static_ndarray_ops::np_dsplit(&mut self.builder, &visited_args)
                }
            }
            (Some("np"), "floor") => crate::ops::static_ndarray_ops::np_floor(&mut self.builder, &visited_args),
            (Some("np"), "ceil") => crate::ops::static_ndarray_ops::np_ceil(&mut self.builder, &visited_args),
            (Some("np"), "trunc") => crate::ops::static_ndarray_ops::np_trunc(&mut self.builder, &visited_args),
            (Some("np"), "round") => crate::ops::static_ndarray_ops::np_round(&mut self.builder, &visited_args),
            (Some("np"), "reciprocal") => crate::ops::static_ndarray_ops::np_reciprocal(&mut self.builder, &visited_args),
            (Some("np"), "where") => crate::ops::static_ndarray_ops::np_where(&mut self.builder, &visited_args),
            (Some("np"), "clip") => crate::ops::static_ndarray_ops::np_clip(&mut self.builder, &visited_args),
            (Some("np"), "mean") => crate::ops::static_ndarray_ops::np_mean(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "var") => crate::ops::static_ndarray_ops::np_var(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "std") => crate::ops::static_ndarray_ops::np_std(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "cumsum") => crate::ops::static_ndarray_ops::np_cumsum(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "cumprod") => crate::ops::static_ndarray_ops::np_cumprod(&mut self.builder, &visited_args, &_visited_kwargs),
            (Some("np"), "promote_to_dynamic") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::promote::promote_static_to_dynamic(&mut self.builder, &val)
            }
            (Some("np"), "block") => {
                // Block depth must be measured from the *AST*, not the
                // visited Value: after Python list literals have been turned
                // into `Value::List`, they are indistinguishable from
                // ndarray axes. Walk the leftmost path of the original AST,
                // counting only Python list/tuple literals.
                fn ast_block_depth(node: &ASTNode) -> usize {
                    match node {
                        ASTNode::ASTSquareBrackets(sb) => {
                            if sb.values.is_empty() { 1 }
                            else { 1 + ast_block_depth(&sb.values[0]) }
                        }
                        ASTNode::ASTParenthesis(p) => {
                            if p.values.is_empty() { 1 }
                            else { 1 + ast_block_depth(&p.values[0]) }
                        }
                        _ => 0,
                    }
                }
                let depth = n.args.first().map(ast_block_depth).unwrap_or(0);
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::np_block_with_depth(&val, depth)
            }
            (Some("np"), "reshape") => {
                // P4c: native StaticArray dispatch via array_ops::reshape fast-path.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::reshape(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    );
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_reshape(&mut self.builder, &val, &visited_args[1..])
            }
            (Some("np"), "repeat") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_repeat(&mut self.builder, &val, &visited_args[1..], &_visited_kwargs)
            }

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
                let v = self.ctx.get(var).unwrap_or(Value::None);
                // For Value::StaticArray, .copy() must clone the underlying
                // segment so subsequent mutations don't alias back to the
                // source. Without this, P3 segment-shared writes would
                // mutate both the original and the copy.
                if let Value::StaticArray { .. } = &v {
                    let lst = crate::helpers::static_array::to_value_list(&mut self.builder, &v);
                    if let Some(sa) = crate::helpers::static_array::to_static_array(&mut self.builder, &lst) {
                        sa
                    } else {
                        lst
                    }
                } else {
                    v
                }
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

            // ── Complex .real / .imag / .conjugate accessors ──────────
            (Some(var), "real")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::Complex { .. })) =>
            {
                if let Some(Value::Complex { real, .. }) = self.ctx.get(var) {
                    Value::Float(real)
                } else {
                    unreachable!()
                }
            }
            (Some(var), "imag")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::Complex { .. })) =>
            {
                if let Some(Value::Complex { imag, .. }) = self.ctx.get(var) {
                    Value::Float(imag)
                } else {
                    unreachable!()
                }
            }
            (Some(var), "conjugate")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::Complex { .. })) =>
            {
                if let Some(Value::Complex { real, imag }) = self.ctx.get(var) {
                    let zero = self.builder.ir_constant_float(0.0);
                    let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag));
                    let ni = match neg_imag {
                        Value::Float(s) => s,
                        _ => unreachable!(),
                    };
                    Value::Complex { real, imag: ni }
                } else {
                    unreachable!()
                }
            }
            // P5a: same accessors on a Complex StaticArray operand.
            (Some(var), "real")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::StaticArray { dtype: crate::types::NumberType::Complex, .. })) =>
            {
                let v = self.ctx.get(var).unwrap();
                crate::helpers::static_array_complex::np_real_static_array(&mut self.builder, &v)
            }
            (Some(var), "imag")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::StaticArray { dtype: crate::types::NumberType::Complex, .. })) =>
            {
                let v = self.ctx.get(var).unwrap();
                crate::helpers::static_array_complex::np_imag_static_array(&mut self.builder, &v)
            }
            (Some(var), "conjugate")
                if self.ctx.exists(var)
                    && matches!(self.ctx.get(var), Some(Value::StaticArray { dtype: crate::types::NumberType::Complex, .. })) =>
            {
                let v = self.ctx.get(var).unwrap();
                crate::helpers::static_array_complex::np_conj_static_array(&mut self.builder, &v)
            }

            // ── DynamicNDArray method dispatch ────────────────────────
            (Some(var), method) if self.ctx.exists(var) && matches!(self.ctx.get(var), Some(Value::DynamicNDArray(_))) => {
                let val = self.ctx.get(var).unwrap();
                self.dispatch_dyn_ndarray_method(val, method, &visited_args, &_visited_kwargs)
            }

            // ── Method calls on expr attributes ────────────────────────
            //
            // Unified ops: each entry point handles static/dynamic internally.
            (Some(var), method @ ("sum" | "any" | "all" | "prod" | "min" | "max")) if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_reduce(
                    &mut self.builder,
                    method,
                    &val,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return out;
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::reduce(&mut self.builder, method, &val, axis_arg)
            }
            (Some(var), "transpose") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let args = if let Some(axes_val) = _visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &val, &args,
                    ) {
                        return out;
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::transpose(&mut self.builder, &val, &args)
            }
            (Some(var), "T") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &val, &[],
                    ) {
                        return out;
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &[])
            }
            (Some(var), "tolist") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val)
            }
            (Some(var), "astype") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                // Determine target type from the argument (int or float class)
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&val, target_float)
            }
            (Some(var), method @ ("argmax" | "argmin")) if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &val,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return out;
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::argmax_argmin(&mut self.builder, &val, axis_arg, method == "argmax")
            }

            // ── NDArray property access ──────────────────────────────
            (Some(var), "shape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
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
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float {
                    Value::Class(ZinniaType::Float)
                } else {
                    Value::Class(ZinniaType::Integer)
                }
            }

            // ── NDArray ndim, size, flatten, flat, reshape, moveaxis, repeat, filter ─
            (Some(var), "ndim") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            (Some(var), "size") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            (Some(var), "flatten") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_flatten(
                        &mut self.builder, &val,
                    ) {
                        return out;
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "flat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            (Some(var), "reshape") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch (delegates to array_ops::reshape
                // which has its own StaticArray fast-path before the legacy fallback).
                if matches!(val, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::reshape(&mut self.builder, &val, &visited_args);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::reshape(&mut self.builder, &val, &visited_args)
            }
            (Some(var), "moveaxis") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch through array_ops::moveaxis.
                if matches!(val, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::moveaxis(&mut self.builder, &val, &visited_args);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::moveaxis(&mut self.builder, &val, &visited_args)
            }
            (Some(var), "repeat") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::repeat(&mut self.builder, &val, &visited_args, &_visited_kwargs)
            }
            (Some(var), "swapaxes") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::swapaxes(&mut self.builder, &val, &visited_args)
            }
            (Some(var), "mean") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_reduce(
                    &mut self.builder,
                    "mean",
                    &val,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return out;
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_mean(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "var") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_var(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "std") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_std(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "cumsum") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_cumsum(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "cumprod") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_cumprod(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "squeeze") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_squeeze(
                        &mut self.builder, &val, axis_arg,
                    ) {
                        return out;
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                let mut all_args: Vec<Value> = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_squeeze(&mut self.builder, &all_args, &_visited_kwargs)
            }
            (Some(var), "filter") if self.ctx.exists(var) => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::filter(&mut self.builder, &val, &visited_args)
            }

            // ── np.* fallback to the np_like registry ───────────────────
            //
            // Anything under the `np` namespace that wasn't matched by an
            // explicit arm above falls through to the registry, where the
            // `define_np_*` macros (now vectorized) handle ops like np.sqrt,
            // np.exp, np.sin, np.abs, np.minimum, np.equal, etc. Positional
            // args are mapped to the macro-expected `x` / `x1` / `x2`
            // kwargs.
            //
            // For unary ops invoked on a composite value, we centrally
            // vectorize by walking leaves and re-dispatching at each one.
            // This fixes hand-rolled ops (np.negative, np.sign,
            // np.positive, np.logical_not, np.abs/absolute/fabs, …) which
            // don't use the auto-vectorizing macro.
            (Some("np"), name) | (Some("zinnia"), name) | (Some("math"), name) => {
                fn dispatch_scalar(
                    builder: &mut crate::builder::IRBuilder,
                    name: &str,
                    kwargs: std::collections::HashMap<String, Value>,
                ) -> Option<Value> {
                    let op_args = crate::ops::OpArgs::new(kwargs);
                    crate::ops::registry::build_op(
                        name,
                        Some(crate::ops::registry::OpNamespace::Np),
                        builder,
                        &op_args,
                    )
                }
                fn vectorize_unary_np(
                    builder: &mut crate::builder::IRBuilder,
                    name: &str,
                    base_kwargs: &std::collections::HashMap<String, Value>,
                    x: &Value,
                ) -> Value {
                    match x {
                        Value::List(d) | Value::Tuple(d) => {
                            let vals: Vec<Value> = d
                                .values
                                .iter()
                                .map(|v| vectorize_unary_np(builder, name, base_kwargs, v))
                                .collect();
                            let types = vals.iter().map(|v| v.zinnia_type()).collect();
                            Value::List(CompositeData {
                                elements_type: types,
                                values: vals,
                            })
                        }
                        Value::DynamicNDArray(d) => {
                            let out_dtype = d.dtype;
                            let name_owned = name.to_string();
                            let kw_clone = base_kwargs.clone();
                            crate::ops::dyn_ndarray::binary::dyn_unary_op(
                                builder, d, out_dtype,
                                move |b, v| {
                                    let mut kw = kw_clone.clone();
                                    kw.insert("x".to_string(), v.clone());
                                    dispatch_scalar(b, &name_owned, kw)
                                        .unwrap_or_else(|| panic!("np.{} is not implemented", name_owned))
                                },
                            )
                        }
                        Value::Complex { real, imag } => {
                            // Complex transcendentals via Euler / standard identities,
                            // built on the existing real Float trig+exp+sqrt gates.
                            let re = Value::Float(real.clone());
                            let im = Value::Float(imag.clone());
                            match name {
                                "exp" => {
                                    // exp(a + bi) = e^a * (cos(b) + i*sin(b))
                                    let exp_a = builder.ir_exp_f(&re);
                                    let cos_b = builder.ir_cos_f(&im);
                                    let sin_b = builder.ir_sin_f(&im);
                                    let r = builder.ir_mul_f(&exp_a, &cos_b);
                                    let i = builder.ir_mul_f(&exp_a, &sin_b);
                                    let r_sv = match r { Value::Float(s) => s, _ => unreachable!() };
                                    let i_sv = match i { Value::Float(s) => s, _ => unreachable!() };
                                    Value::Complex { real: r_sv, imag: i_sv }
                                }
                                "sin" => {
                                    // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
                                    let sa = builder.ir_sin_f(&re);
                                    let ca = builder.ir_cos_f(&re);
                                    let cb = builder.ir_cosh_f(&im);
                                    let sb = builder.ir_sinh_f(&im);
                                    let r = builder.ir_mul_f(&sa, &cb);
                                    let i = builder.ir_mul_f(&ca, &sb);
                                    let r_sv = match r { Value::Float(s) => s, _ => unreachable!() };
                                    let i_sv = match i { Value::Float(s) => s, _ => unreachable!() };
                                    Value::Complex { real: r_sv, imag: i_sv }
                                }
                                "cos" => {
                                    // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
                                    let ca = builder.ir_cos_f(&re);
                                    let sa = builder.ir_sin_f(&re);
                                    let cb = builder.ir_cosh_f(&im);
                                    let sb = builder.ir_sinh_f(&im);
                                    let r = builder.ir_mul_f(&ca, &cb);
                                    let prod_sa_sb = builder.ir_mul_f(&sa, &sb);
                                    let zero = builder.ir_constant_float(0.0);
                                    let i_neg = builder.ir_sub_f(&zero, &prod_sa_sb);
                                    let r_sv = match r { Value::Float(s) => s, _ => unreachable!() };
                                    let i_sv = match i_neg { Value::Float(s) => s, _ => unreachable!() };
                                    Value::Complex { real: r_sv, imag: i_sv }
                                }
                                "sqrt" => {
                                    // sqrt(z) where z = a + bi:
                                    // |z| = sqrt(a² + b²); when im == 0 and re ≥ 0, fall back
                                    // to real sqrt with imag=0.
                                    // Polar form: sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
                                    // Implemented via the formula:
                                    //   r = sqrt((|z| + a) / 2)
                                    //   s = sign(b) * sqrt((|z| - a) / 2)
                                    let aa = builder.ir_mul_f(&re, &re);
                                    let bb = builder.ir_mul_f(&im, &im);
                                    let mod_sq = builder.ir_add_f(&aa, &bb);
                                    let modulus = builder.ir_sqrt_f(&mod_sq);
                                    let two = builder.ir_constant_float(2.0);
                                    let mod_plus_a = builder.ir_add_f(&modulus, &re);
                                    let mod_minus_a = builder.ir_sub_f(&modulus, &re);
                                    let half_plus = builder.ir_div_f(&mod_plus_a, &two);
                                    let half_minus = builder.ir_div_f(&mod_minus_a, &two);
                                    let r = builder.ir_sqrt_f(&half_plus);
                                    let s_abs = builder.ir_sqrt_f(&half_minus);
                                    // Sign of imag: if b >= 0 then +s_abs else -s_abs.
                                    let zero = builder.ir_constant_float(0.0);
                                    let pos = builder.ir_greater_than_or_equal_f(&im, &zero);
                                    let neg_s = builder.ir_sub_f(&zero, &s_abs);
                                    let s = builder.ir_select_f(&pos, &s_abs, &neg_s);
                                    let r_sv = match r { Value::Float(sv) => sv, _ => unreachable!() };
                                    let s_sv = match s { Value::Float(sv) => sv, _ => unreachable!() };
                                    Value::Complex { real: r_sv, imag: s_sv }
                                }
                                "log" => {
                                    // log(a + bi) = log(|z|) + i*arg(z) where arg = atan2(b, a)
                                    // Skip until atan2 lands on the real path; fall through
                                    // to scalar dispatch which will panic.
                                    let mut kw = base_kwargs.clone();
                                    kw.insert("x".to_string(), x.clone());
                                    dispatch_scalar(builder, name, kw)
                                        .unwrap_or_else(|| panic!("np.{} on Complex is not yet implemented", name))
                                }
                                _ => {
                                    panic!("np.{} on Complex is not yet implemented (compiler.complex-transcendentals scope)", name)
                                }
                            }
                        }
                        _ => {
                            let mut kw = base_kwargs.clone();
                            kw.insert("x".to_string(), x.clone());
                            dispatch_scalar(builder, name, kw)
                                .unwrap_or_else(|| panic!("np.{} is not implemented", name))
                        }
                    }
                }

                // Mapping for binary ops to the short names accepted by
                // `apply_binary_op`. Used for centralized binary
                // vectorization below.
                let binary_apply_name: Option<&str> = match member {
                    "add" => Some("add"),
                    "subtract" => Some("sub"),
                    "multiply" => Some("mul"),
                    "divide" => Some("div"),
                    "floor_divide" => Some("floor_div"),
                    "mod" | "fmod" => Some("mod"),
                    "power" | "pow" => Some("pow"),
                    "equal" => Some("eq"),
                    "not_equal" => Some("ne"),
                    "less" => Some("lt"),
                    "less_equal" => Some("lte"),
                    "greater" => Some("gt"),
                    "greater_equal" => Some("gte"),
                    "logical_and" => Some("and"),
                    "logical_or" => Some("or"),
                    _ => None,
                };

                // P5a: np.abs/absolute/fabs on a Complex StaticArray → fresh
                // Float StaticArray. Other np.* unary ops on Complex
                // StaticArray fall through to legacy path via deep_to_value_list
                // (for now; native vectorization is a follow-up).
                let result = if visited_args.len() == 1
                    && matches!(visited_args_orig.first(), Some(Value::StaticArray { dtype: crate::types::NumberType::Complex, .. }))
                    && matches!(member, "abs" | "absolute" | "fabs")
                {
                    let arr = visited_args_orig[0].clone();
                    Some(crate::helpers::static_array_complex::np_abs_complex_static_array(
                        &mut self.builder, &arr,
                    ))
                } else if visited_args.len() == 1
                    && matches!(visited_args[0], Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_) | Value::Complex { .. })
                {
                    // Centralized unary vectorization.
                    Some(vectorize_unary_np(
                        &mut self.builder,
                        member,
                        &_visited_kwargs,
                        &visited_args[0],
                    ))
                } else if visited_args.len() == 2
                    && (matches!(visited_args[0], Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_))
                        || matches!(visited_args[1], Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_)))
                    && binary_apply_name.is_some()
                {
                    // Centralized binary vectorization for composite operands.
                    // Falls through to `apply_binary_op` which already does
                    // shape-broadcasting and dtype promotion.
                    Some(crate::helpers::value_ops::apply_binary_op(
                        &mut self.builder,
                        binary_apply_name.unwrap(),
                        &visited_args[0],
                        &visited_args[1],
                    ))
                } else {
                    let mut all_kwargs = _visited_kwargs.clone();
                    if visited_args.len() == 1 {
                        all_kwargs
                            .entry("x".to_string())
                            .or_insert_with(|| visited_args[0].clone());
                    } else {
                        for (i, v) in visited_args.iter().enumerate() {
                            let key = format!("x{}", i + 1);
                            all_kwargs.entry(key).or_insert_with(|| v.clone());
                        }
                    }
                    dispatch_scalar(&mut self.builder, member, all_kwargs)
                };
                match result {
                    Some(v) => v,
                    None => panic!("np.{} is not implemented", name),
                }
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
                if target.is_none() {
                    // Bare name call like `foo(...)` that didn't match any
                    // builtin / chip / external. Most common cause is a
                    // user-defined helper that isn't decorated with @zk_chip.
                    let mut chip_names: Vec<&str> =
                        self.registered_chips.keys().map(|s| s.as_str()).collect();
                    chip_names.sort();
                    let chips_hint = if chip_names.is_empty() {
                        "no @zk_chip helpers are registered in this module".to_string()
                    } else {
                        format!("registered chips: {}", chip_names.join(", "))
                    };
                    panic!(
                        "Unknown function `{}` in @zk_circuit. If `{}` is a helper you defined, decorate it with @zk_chip (or @zk_external for off-circuit witness functions). {}",
                        member, member, chips_hint
                    )
                }
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

        // P1 segarr boundary: convert StaticArray target / args to legacy List
        // view before legacy method dispatch. See `visit_named_attr` for the
        // explanation.
        //
        // P4b: keep the un-converted target / args / kwargs around so the
        // migrated reduction entry points can route through
        // `static_array_reductions` directly on the StaticArray
        // representation.
        let target_orig = target.clone();
        let visited_args_orig: Vec<Value> = visited_args.clone();
        let visited_kwargs_orig: HashMap<String, Value> = visited_kwargs.clone();
        let target = crate::helpers::static_array::deep_to_value_list(&mut self.builder, &target);
        let visited_args: Vec<Value> = visited_args
            .iter()
            .map(|v| crate::helpers::static_array::deep_to_value_list(&mut self.builder, v))
            .collect();
        let visited_kwargs: HashMap<String, Value> = visited_kwargs
            .into_iter()
            .map(|(k, v)| (k, crate::helpers::static_array::deep_to_value_list(&mut self.builder, &v)))
            .collect();

        // DynamicNDArray dispatch — route to dedicated handler
        if matches!(target, Value::DynamicNDArray(_)) {
            return self.dispatch_dyn_ndarray_method(
                target, n.member.as_str(), &visited_args, &visited_kwargs,
            );
        }

        // Complex .real / .imag / .conjugate accessors on any expression.
        if let Value::Complex { real, imag } = &target {
            match n.member.as_str() {
                "real" => return Value::Float(real.clone()),
                "imag" => return Value::Float(imag.clone()),
                "conjugate" => {
                    let zero = self.builder.ir_constant_float(0.0);
                    let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag.clone()));
                    let ni = match neg_imag {
                        Value::Float(s) => s,
                        _ => unreachable!(),
                    };
                    return Value::Complex { real: real.clone(), imag: ni };
                }
                _ => {}
            }
        }

        match n.member.as_str() {
            method @ ("sum" | "any" | "all" | "prod" | "min" | "max" | "mean") => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_reduce(
                    &mut self.builder,
                    method,
                    &target_orig,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return out;
                }
                if method == "mean" {
                    let mut all_args = vec![target.clone()];
                    all_args.extend(visited_args.iter().cloned());
                    return crate::ops::static_ndarray_ops::np_mean(
                        &mut self.builder,
                        &all_args,
                        &visited_kwargs,
                    );
                }
                let axis_arg = visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::reduce(&mut self.builder, method, &target, axis_arg)
            }
            method @ ("argmax" | "argmin") => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = visited_kwargs_orig
                    .get("keepdims")
                    .and_then(|v| v.bool_val())
                    .unwrap_or(false);
                if let Some(out) = crate::helpers::static_array_reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &target_orig,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return out;
                }
                let axis_arg = visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::argmax_argmin(&mut self.builder, &target, axis_arg, method == "argmax")
            }
            "transpose" => {
                let args = if let Some(axes_val) = visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &target_orig, &args,
                    ) {
                        return out;
                    }
                }
                crate::helpers::array_ops::transpose(&mut self.builder, &target, &args)
            }
            "T" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &target_orig, &[],
                    ) {
                        return out;
                    }
                }
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &target, &[])
            }
            "tolist" => target,
            "astype" => {
                let target_float = if let Some(Value::Class(ZinniaType::Float)) = visited_args.first() {
                    true
                } else {
                    false
                };
                self.cast_composite(&target, target_float)
            }
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
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_flatten(
                        &mut self.builder, &target_orig,
                    ) {
                        return out;
                    }
                }
                let flat = crate::helpers::composite::flatten_composite(&target);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: flat })
            }
            "reshape" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::reshape(&mut self.builder, &target_orig, &visited_args);
                }
                crate::helpers::array_ops::reshape(&mut self.builder, &target, &visited_args)
            }
            "moveaxis" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::moveaxis(&mut self.builder, &target_orig, &visited_args);
                }
                crate::helpers::array_ops::moveaxis(&mut self.builder, &target, &visited_args)
            }
            "repeat" => crate::helpers::array_ops::repeat(&mut self.builder, &target, &visited_args, &visited_kwargs),
            "filter" => crate::helpers::array_ops::filter(&mut self.builder, &target, &visited_args),
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
