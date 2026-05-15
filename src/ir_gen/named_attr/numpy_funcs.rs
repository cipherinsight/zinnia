//! Numpy / math / zinnia / DynamicNDArray namespaced function dispatch.
//!
//! Split into two halves:
//!   - `try_numpy_func`: explicit named-arm matches for `np.zeros`,
//!     `np.concatenate`, `np.split`, etc. plus `DynamicNDArray.zeros` /
//!     `zinnia.zeros` style class-method calls.
//!   - `try_np_fallback`: the catch-all `(Some("np"|"zinnia"|"math"), name)`
//!     arm that delegates to the `ops::registry` and the centralised
//!     unary/binary vectorisation paths.

use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, ScalarValue, Value, ValueId, ZinniaType};

use super::DispatchCtx;
use super::super::IRGenerator;

/// Check if the first arg is a list/tuple containing DynamicNDArrays.
fn has_dynamic_array_in_list(args: &[Value]) -> bool {
    match args.first() {
        Some(Value::List(cd)) | Some(Value::Tuple(cd)) => {
            cd.values.iter().any(|v| matches!(v, Value::DynamicNDArray(_)))
        }
        _ => false,
    }
}

/// P4c: helper to detect StaticArray inputs in a list/tuple — used to
/// decide whether to route concatenate/stack/etc. through the native
/// path before falling back to the legacy nested-List handlers.
fn list_contains_static_array(arg: &Value) -> bool {
    match arg {
        Value::List(cd) | Value::Tuple(cd) => {
            cd.values.iter().any(|v| matches!(v, Value::StaticArray { .. }))
        }
        _ => false,
    }
}

impl IRGenerator {
    pub(crate) fn try_numpy_func(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let visited_args = ctx.args;
        let _visited_kwargs = ctx.kwargs;
        let visited_args_orig = ctx.args_orig;
        let visited_kwargs_orig = ctx.kwargs_orig;
        let target = ctx.target?;
        let member = ctx.member;

        let v = match (target, member) {
            // ── np.* (numpy-like operations) ───────────────────────────
            // numpy dtype aliases — collapse width into ZinniaType (Float / Integer / Boolean)
            ("np", "float16" | "float32" | "float64" | "double" | "single" | "half" | "longdouble") => Value::Class(ZinniaType::Float),
            ("np", "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
                 | "intp" | "uintp" | "long" | "byte" | "short" | "intc" | "uintc" | "ubyte"
                 | "ushort" | "longlong" | "ulonglong") => Value::Class(ZinniaType::Integer),
            ("np", "bool_") => Value::Class(ZinniaType::Boolean),
            ("np", "complex64" | "complex128" | "complex256" | "csingle" | "cdouble" | "clongdouble") => Value::Class(ZinniaType::Complex),
            // Complex helpers
            ("np", "conj" | "conjugate") => {
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
            ("np", "real") => {
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
            ("np", "imag") => {
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
            ("np", "pi") | ("math", "pi") => self.builder.ir_constant_float(std::f64::consts::PI),
            ("np", "e") | ("math", "e") => self.builder.ir_constant_float(std::f64::consts::E),
            ("np", "asarray" | "array") => {
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
            ("np", "zeros") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, visited_args, _visited_kwargs, 0),
            ("np", "ones") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, visited_args, _visited_kwargs, 1),
            ("np", "empty" | "ndarray") => crate::ops::static_ndarray_ops::np_fill(&mut self.builder, visited_args, _visited_kwargs, 0),
            ("np", "zeros_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, visited_args, _visited_kwargs, 0),
            ("np", "ones_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, visited_args, _visited_kwargs, 1),
            ("np", "empty_like") => crate::ops::static_ndarray_ops::np_fill_like(&mut self.builder, visited_args, _visited_kwargs, 0),
            ("np", "identity" | "eye") => crate::ops::static_ndarray_ops::np_identity(&mut self.builder, visited_args),
            // numpy reduction aliases — forward to the existing reduce/argmax_argmin path used by x.sum() / x.argmin()
            ("np", method @ ("sum" | "prod" | "min" | "max" | "any" | "all")) => {
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
                    return Some(out);
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.get(1));
                crate::helpers::array_ops::reduce(&mut self.builder, method, &val, axis_arg)
            }
            ("np", method @ ("argmax" | "argmin")) => {
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
                    return Some(out);
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.get(1));
                crate::helpers::array_ops::argmax_argmin(&mut self.builder, &val, axis_arg, method == "argmax")
            }
            ("np", "dot" | "matmul") => {
                let lhs = visited_args.first().cloned().unwrap_or(Value::None);
                let rhs = visited_args.get(1).cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::matmul(&mut self.builder, &lhs, &rhs)
            }
            ("np", "square") => crate::ops::static_ndarray_ops::np_square(&mut self.builder, visited_args),
            ("np", "diff") => crate::ops::static_ndarray_ops::np_diff(&mut self.builder, visited_args),
            ("np", "outer") => crate::ops::static_ndarray_ops::np_outer(&mut self.builder, visited_args),
            ("np", "add_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "add"),
            ("np", "subtract_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "sub"),
            ("np", "multiply_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "mul"),
            ("np", "divide_outer") => crate::ops::static_ndarray_ops::np_outer_op(&mut self.builder, visited_args, "div"),
            ("np", "arange") => crate::ops::static_ndarray_ops::np_arange(&mut self.builder, visited_args),
            ("np", "linspace") => crate::ops::static_ndarray_ops::np_linspace(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "allclose") => crate::ops::static_ndarray_ops::np_allclose(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "concatenate") => {
                if has_dynamic_array_in_list(visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(&mut self.builder, visited_args, _visited_kwargs)
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
                                return Some(out);
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_concatenate(&mut self.builder, visited_args, _visited_kwargs)
                }
            }
            ("np", "stack") => {
                if has_dynamic_array_in_list(visited_args) {
                    crate::ops::dyn_ndarray::memory_ops::dyn_stack(&mut self.builder, visited_args, _visited_kwargs)
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
                                return Some(out);
                            }
                        }
                    }
                    crate::ops::static_ndarray_ops::np_stack(&mut self.builder, visited_args, _visited_kwargs)
                }
            }
            ("np", "vstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_vstack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_vstack(&mut self.builder, visited_args)
            }
            ("np", "row_stack") => crate::ops::static_ndarray_ops::np_row_stack(&mut self.builder, visited_args),
            ("np", "hstack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_hstack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_hstack(&mut self.builder, visited_args)
            }
            ("np", "dstack") => crate::ops::static_ndarray_ops::np_dstack(&mut self.builder, visited_args),
            ("np", "column_stack") => {
                // P4c: native StaticArray dispatch.
                if let Some(arrays_arg) = visited_args_orig.first() {
                    if list_contains_static_array(arrays_arg) {
                        if let Some(out) = crate::helpers::static_array_shape::try_apply_column_stack(
                            &mut self.builder, arrays_arg,
                        ) {
                            return Some(out);
                        }
                    }
                }
                crate::ops::static_ndarray_ops::np_column_stack(&mut self.builder, visited_args)
            }
            ("np", "swapaxes") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_swapaxes(&mut self.builder, &val, &visited_args[1..])
            }
            ("np", "moveaxis") => {
                // P4c: native StaticArray dispatch via array_ops::moveaxis fast-path,
                // using the un-converted orig args.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    return Some(crate::helpers::array_ops::moveaxis(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    ));
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_moveaxis(&mut self.builder, &val, &visited_args[1..])
            }
            ("np", "transpose") => {
                // P4c: native StaticArray dispatch.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_transpose(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    ) {
                        return Some(out);
                    }
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &visited_args[1..])
            }
            ("np", "flip") => crate::ops::static_ndarray_ops::np_flip(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "flipud") => crate::ops::static_ndarray_ops::np_flipud(&mut self.builder, visited_args),
            ("np", "fliplr") => crate::ops::static_ndarray_ops::np_fliplr(&mut self.builder, visited_args),
            ("np", "rot90") => crate::ops::static_ndarray_ops::np_rot90(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "squeeze") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_kwargs_orig.get("axis").or_else(|| visited_args_orig.get(1));
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_squeeze(
                        &mut self.builder, &val_orig, axis_arg,
                    ) {
                        return Some(out);
                    }
                }
                crate::ops::static_ndarray_ops::np_squeeze(&mut self.builder, visited_args, _visited_kwargs)
            }
            ("np", "expand_dims") => {
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    let axis_arg = visited_args_orig.get(1);
                    if let Some(out) = crate::helpers::static_array_shape::try_apply_expand_dims(
                        &mut self.builder, &val_orig, axis_arg,
                    ) {
                        return Some(out);
                    }
                }
                crate::ops::static_ndarray_ops::np_expand_dims(&mut self.builder, visited_args)
            }
            ("np", "broadcast_to") => crate::ops::static_ndarray_ops::np_broadcast_to(&mut self.builder, visited_args),
            ("np", "atleast_1d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 1),
            ("np", "atleast_2d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 2),
            ("np", "atleast_3d") => crate::ops::static_ndarray_ops::np_atleast_nd(&mut self.builder, visited_args, 3),
            ("np", "tile") => crate::ops::static_ndarray_ops::np_tile(&mut self.builder, visited_args),
            ("np", "take") => {
                // np.take(arr, indices, axis=None). For now we support the
                // default-axis case on a 1-D source, or explicit axis=0 — the
                // semantics that compose with `fancy_index_static`. Multi-D
                // flatten-then-take and arbitrary axes are TBD.
                let arr = visited_args.first().cloned().unwrap_or(Value::None);
                let indices = visited_args.get(1).cloned().unwrap_or(Value::None);
                let axis_raw = _visited_kwargs.get("axis")
                    .or_else(|| visited_args.get(2));
                let axis_is_default = match axis_raw {
                    None => true,
                    Some(Value::None) => true,
                    _ => false,
                };
                let axis_val = axis_raw.and_then(|v| v.int_val()).unwrap_or(0);
                let data = match &arr {
                    Value::List(d) | Value::Tuple(d) => d.clone(),
                    _ => panic!("np.take: first argument must be a static-shape array"),
                };
                // Default-axis path on a 1-D source matches axis=0; for
                // multi-D `axis=None` we'd need to flatten first.
                let is_1d = data.values.iter().all(|v| !matches!(v, Value::List(_) | Value::Tuple(_)));
                if !axis_is_default && axis_val != 0 {
                    panic!("np.take with axis != 0 is not yet implemented");
                }
                if axis_is_default && !is_1d {
                    panic!("np.take on multi-dimensional arrays with axis=None (flatten) is not yet implemented");
                }
                let out = match crate::helpers::ndarray::fancy_index_static(&data, &indices) {
                    Ok(v) => v,
                    Err(msg) => panic!("np.take: {}", msg),
                };
                // Group 5c (E-axis): content fact relay.
                if let (Some(in_vid), Some(out_vid)) = (arr.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(
                        &mut self.builder, in_vid, out_vid,
                    );
                }
                out
            }
            ("np", "split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], _visited_kwargs)
                } else {
                    crate::ops::static_ndarray_ops::np_split(&mut self.builder, visited_args, _visited_kwargs)
                }
            }
            ("np", "array_split") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    crate::ops::dyn_ndarray::memory_ops::dyn_array_split(&mut self.builder, &data, &visited_args[1..], _visited_kwargs)
                } else {
                    crate::ops::static_ndarray_ops::np_array_split(&mut self.builder, visited_args, _visited_kwargs)
                }
            }
            ("np", "hsplit") => {
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
                    crate::ops::static_ndarray_ops::np_hsplit(&mut self.builder, visited_args)
                }
            }
            ("np", "vsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert("axis".to_string(), Value::Integer(ScalarValue::new(Some(0), None)));
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &kw)
                } else {
                    crate::ops::static_ndarray_ops::np_vsplit(&mut self.builder, visited_args)
                }
            }
            ("np", "dsplit") => {
                if matches!(visited_args.first(), Some(Value::DynamicNDArray(_))) {
                    let data = match &visited_args[0] {
                        Value::DynamicNDArray(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let mut kw = _visited_kwargs.clone();
                    kw.insert("axis".to_string(), Value::Integer(ScalarValue::new(Some(2), None)));
                    crate::ops::dyn_ndarray::memory_ops::dyn_split(&mut self.builder, &data, &visited_args[1..], &kw)
                } else {
                    crate::ops::static_ndarray_ops::np_dsplit(&mut self.builder, visited_args)
                }
            }
            ("np", "floor") => crate::ops::static_ndarray_ops::np_floor(&mut self.builder, visited_args),
            ("np", "ceil") => crate::ops::static_ndarray_ops::np_ceil(&mut self.builder, visited_args),
            ("np", "trunc") => crate::ops::static_ndarray_ops::np_trunc(&mut self.builder, visited_args),
            ("np", "round") => crate::ops::static_ndarray_ops::np_round(&mut self.builder, visited_args),
            ("np", "reciprocal") => crate::ops::static_ndarray_ops::np_reciprocal(&mut self.builder, visited_args),
            ("np", "where") => crate::ops::static_ndarray_ops::np_where(&mut self.builder, visited_args),
            ("np", "clip") => crate::ops::static_ndarray_ops::np_clip(&mut self.builder, visited_args),
            ("np", "mean") => crate::ops::static_ndarray_ops::np_mean(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "var") => crate::ops::static_ndarray_ops::np_var(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "std") => crate::ops::static_ndarray_ops::np_std(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "cumsum") => crate::ops::static_ndarray_ops::np_cumsum(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "cumprod") => crate::ops::static_ndarray_ops::np_cumprod(&mut self.builder, visited_args, _visited_kwargs),
            ("np", "promote_to_dynamic") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::helpers::promote::promote_static_to_dynamic(&mut self.builder, &val)
            }
            ("np", "block") => {
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
                let depth = ctx.ast_node.args.first().map(ast_block_depth).unwrap_or(0);
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::np_block_with_depth(&val, depth)
            }
            ("np", "reshape") => {
                // P4c: native StaticArray dispatch via array_ops::reshape fast-path.
                let val_orig = visited_args_orig.first().cloned().unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    return Some(crate::helpers::array_ops::reshape(
                        &mut self.builder, &val_orig, &visited_args_orig[1..],
                    ));
                }
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_reshape(&mut self.builder, &val, &visited_args[1..])
            }
            ("np", "repeat") => {
                let val = visited_args.first().cloned().unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::ndarray_repeat(&mut self.builder, &val, &visited_args[1..], _visited_kwargs)
            }

            // ── DynamicNDArray class methods ────────────────────────────
            ("DynamicNDArray", "zeros") | ("zinnia", "zeros") => {
                crate::ops::dyn_ndarray::metadata::dyn_zeros(&mut self.builder, visited_args, _visited_kwargs)
            }
            ("DynamicNDArray", "ones") | ("zinnia", "ones") => {
                crate::ops::dyn_ndarray::metadata::dyn_ones(&mut self.builder, visited_args, _visited_kwargs)
            }
            ("DynamicNDArray", "eye") | ("zinnia", "eye") => {
                crate::ops::dyn_ndarray::constructors::dyn_eye(&mut self.builder, visited_args, _visited_kwargs)
            }
            ("DynamicNDArray", "concatenate") | ("zinnia", "concatenate") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_concatenate(&mut self.builder, visited_args, _visited_kwargs)
            }
            ("DynamicNDArray", "stack") | ("zinnia", "stack") => {
                crate::ops::dyn_ndarray::memory_ops::dyn_stack(&mut self.builder, visited_args, _visited_kwargs)
            }
            _ => return None,
        };
        Some(v)
    }

    /// np.* fallback to the `np_like` registry. Anything under the `np` /
    /// `zinnia` / `math` namespace that wasn't matched by an explicit arm
    /// in `try_numpy_func` (or one of the explicit list / ndarray method
    /// arms in between) lands here. The registry plus centralised
    /// unary/binary vectorisation handles ops like np.sqrt, np.exp,
    /// np.sin, np.abs, np.minimum, np.equal, etc.
    pub(crate) fn try_np_fallback(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let target = ctx.target?;
        if !matches!(target, "np" | "zinnia" | "math") {
            return None;
        }
        let name = ctx.member;
        let member = ctx.member;
        let visited_args = ctx.args;
        let visited_args_orig = ctx.args_orig;
        let _visited_kwargs = ctx.kwargs;

        fn dispatch_scalar(
            builder: &mut crate::builder::IRBuilder,
            name: &str,
            kwargs: HashMap<String, Value>,
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
            base_kwargs: &HashMap<String, Value>,
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

                        value_id: ValueId::next(),
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
                _visited_kwargs,
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
            Some(v) => Some(v),
            None => panic!("np.{} is not implemented", name),
        }
    }
}
