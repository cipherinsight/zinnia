//! Dispatch for ndarray-style method calls and property access on a
//! variable: `var.sum()`, `var.shape`, `var.T`, `var.flatten()`, etc.
//! Also covers Complex `.real` / `.imag` / `.conjugate` accessors (both on
//! scalar Complex and Complex `StaticArray` operands), and the
//! `DynamicNDArray` method routing dispatched via
//! `dispatch_dyn_ndarray_method`.
//!
//! Returns `None` when the (target, member) pair does not name a known
//! variable, the variable's runtime type does not match, or the method is
//! not handled by an ndarray-style arm — so the dispatcher chain can fall
//! through to the np.* registry fallback.

use crate::types::{CompositeData, Value, ValueId, ZinniaType};

use super::super::IRGenerator;
use super::DispatchCtx;

impl IRGenerator {
    pub(crate) fn try_ndarray_method(&mut self, ctx: &DispatchCtx) -> Option<Value> {
        let var = ctx.target?;
        if !self.ctx.exists(var) {
            return None;
        }
        let visited_args = ctx.args;
        let _visited_kwargs = ctx.kwargs;
        let visited_args_orig = ctx.args_orig;
        let visited_kwargs_orig = ctx.kwargs_orig;
        let member = ctx.member;

        // ── Complex .real / .imag / .conjugate accessors ──────────
        if matches!(self.ctx.get(var), Some(Value::Complex { .. })) {
            match member {
                "real" => {
                    if let Some(Value::Complex { real, .. }) = self.ctx.get(var) {
                        return Some(Value::Float(real));
                    } else {
                        unreachable!()
                    }
                }
                "imag" => {
                    if let Some(Value::Complex { imag, .. }) = self.ctx.get(var) {
                        return Some(Value::Float(imag));
                    } else {
                        unreachable!()
                    }
                }
                "conjugate" => {
                    if let Some(Value::Complex { real, imag }) = self.ctx.get(var) {
                        let zero = self.builder.ir_constant_float(0.0);
                        let neg_imag = self.builder.ir_sub_f(&zero, &Value::Float(imag));
                        let ni = match neg_imag {
                            Value::Float(s) => s,
                            _ => unreachable!(),
                        };
                        return Some(Value::Complex { real, imag: ni });
                    } else {
                        unreachable!()
                    }
                }
                _ => {}
            }
        }

        if let Some(v @ (Value::Integer(_) | Value::Float(_) | Value::Boolean(_))) =
            self.ctx.get(var)
        {
            match member {
                "real" | "conjugate" => return Some(v),
                "imag" => {
                    return Some(match v {
                        Value::Float(_) => self.builder.ir_constant_float(0.0),
                        _ => self.builder.ir_constant_int(0),
                    })
                }
                _ => {}
            }
        }

        // P5a: same accessors on a Complex StaticArray operand.
        if matches!(
            self.ctx.get(var),
            Some(Value::StaticArray {
                dtype: crate::types::NumberType::Complex,
                ..
            })
        ) {
            match member {
                "real" => {
                    let v = self.ctx.get(var).unwrap();
                    return Some(crate::helpers::static_array::complex::np_real_static_array(
                        &mut self.builder,
                        &v,
                    ));
                }
                "imag" => {
                    let v = self.ctx.get(var).unwrap();
                    return Some(crate::helpers::static_array::complex::np_imag_static_array(
                        &mut self.builder,
                        &v,
                    ));
                }
                "conjugate" => {
                    let v = self.ctx.get(var).unwrap();
                    return Some(crate::helpers::static_array::complex::np_conj_static_array(
                        &mut self.builder,
                        &v,
                    ));
                }
                _ => {}
            }
        }

        if matches!(member, "sum" | "prod" | "min" | "max" | "any" | "all") {
            let mut args = vec![self.ctx.get(var).unwrap()]; args.extend(visited_args_orig.iter().cloned());
            return Some(super::array_compat::reduction(&mut self.builder,member,&args,visited_kwargs_orig));
        }

        // ── DynamicNDArray method dispatch ────────────────────────
        if matches!(self.ctx.get(var), Some(Value::DynamicNDArray(_))) {
            let val = self.ctx.get(var).unwrap();
            return Some(self.dispatch_dyn_ndarray_method(
                val,
                member,
                visited_args,
                _visited_kwargs,
            ));
        }

        // ── Method calls on expr attributes ────────────────────────
        //
        // Unified ops: each entry point handles static/dynamic internally.
        let v = match member {
            "squeeze" => {
                let val=self.ctx.get(var).unwrap();
                crate::helpers::static_array::shape::try_apply_squeeze(&mut self.builder,&val,visited_kwargs_orig.get("axis").or_else(||visited_args_orig.first())).expect("squeeze requires array")
            }
            "clip" => {
                let mut args = vec![self.ctx.get(var).unwrap()];
                args.extend(visited_args_orig.iter().cloned());
                super::array_compat::clip(&mut self.builder, &args, visited_kwargs_orig)
            }
            "sum" | "any" | "all" | "prod" | "min" | "max" => {
                let method = member;
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_reduce(
                    &mut self.builder,
                    method,
                    &val,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return Some(out);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::reduce(&mut self.builder, method, &val, axis_arg)
            }
            "transpose" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let args = if let Some(axes_val) = _visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.to_vec()
                };
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_transpose(
                        &mut self.builder,
                        &val,
                        &args,
                    ) {
                        return Some(out);
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::transpose(&mut self.builder, &val, &args)
            }
            "T" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_transpose(
                        &mut self.builder,
                        &val,
                        &[],
                    ) {
                        return Some(out);
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &val, &[])
            }
            "tolist" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val)
            }
            "astype" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let dtype = visited_args_orig.first().or_else(|| visited_kwargs_orig.get("dtype"))
                    .expect("astype requires a dtype");
                super::array_compat::astype(&mut self.builder, &val, dtype)
            }
            "argmax" | "argmin" => {
                let method = member;
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &val,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return Some(out);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::argmax_argmin(
                    &mut self.builder,
                    &val,
                    axis_arg,
                    method == "argmax",
                )
            }

            // ── NDArray property access ──────────────────────────────
            "shape" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                let shape_vals: Vec<Value> = shape
                    .iter()
                    .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                    .collect();
                let types = vec![ZinniaType::Integer; shape_vals.len()];
                Value::Tuple(CompositeData {
                    elements_type: types,
                    values: shape_vals,
                    value_id: ValueId::next(),
                })
            }
            "dtype" => {
                // Infer dtype from element types
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let flat = crate::helpers::composite::flatten_composite(&val);
                let has_float = flat.iter().any(|v| matches!(v, Value::Float(_)));
                if has_float {
                    Value::Class(ZinniaType::Float)
                } else {
                    Value::Class(ZinniaType::Integer)
                }
            }

            // ── NDArray ndim, size, flatten, flat, reshape, moveaxis, repeat, filter ─
            "ndim" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                self.builder.ir_constant_int(shape.len() as i64)
            }
            "size" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let shape = crate::helpers::composite::get_composite_shape(&val);
                let total: usize = shape.iter().product();
                self.builder.ir_constant_int(total as i64)
            }
            "flatten" => {
                let val_orig = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_flatten(
                        &mut self.builder,
                        &val_orig,
                    ) {
                        return Some(out);
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::base::deep_to_value_list(
                    &mut self.builder,
                    &val_orig,
                );
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                let out = Value::List(CompositeData {
                    elements_type: types,
                    values: flat,
                    value_id: ValueId::next(),
                });
                if let (Some(in_vid), Some(out_vid)) = (val_orig.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(
                        &mut self.builder,
                        in_vid,
                        out_vid,
                    );
                }
                out
            }
            "flat" => {
                let val_orig = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val = crate::helpers::static_array::base::deep_to_value_list(
                    &mut self.builder,
                    &val_orig,
                );
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                let out = Value::List(CompositeData {
                    elements_type: types,
                    values: flat,
                    value_id: ValueId::next(),
                });
                if let (Some(in_vid), Some(out_vid)) = (val_orig.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(
                        &mut self.builder,
                        in_vid,
                        out_vid,
                    );
                }
                out
            }
            "reshape" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch (delegates to array_ops::reshape
                // which has its own StaticArray fast-path before the legacy fallback).
                if matches!(val, Value::StaticArray { .. }) {
                    return Some(crate::helpers::array_ops::reshape(
                        &mut self.builder,
                        &val,
                        visited_args,
                    ));
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::reshape(&mut self.builder, &val, visited_args)
            }
            "moveaxis" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch through array_ops::moveaxis.
                if matches!(val, Value::StaticArray { .. }) {
                    return Some(crate::helpers::array_ops::moveaxis(
                        &mut self.builder,
                        &val,
                        visited_args,
                    ));
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::moveaxis(&mut self.builder, &val, visited_args)
            }
            "repeat" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::repeat(
                    &mut self.builder,
                    &val,
                    visited_args,
                    _visited_kwargs,
                )
            }
            "ravel" => {
                let val_orig = self.ctx.get(var).unwrap_or(Value::None);
                if matches!(val_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_flatten(
                        &mut self.builder,
                        &val_orig,
                    ) {
                        return Some(out);
                    }
                }
                let val = crate::helpers::static_array::base::deep_to_value_list(
                    &mut self.builder,
                    &val_orig,
                );
                let flat = crate::helpers::composite::flatten_composite(&val);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                let out = Value::List(CompositeData {
                    elements_type: types,
                    values: flat,
                    value_id: ValueId::next(),
                });
                if let (Some(in_vid), Some(out_vid)) = (val_orig.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(
                        &mut self.builder,
                        in_vid,
                        out_vid,
                    );
                }
                out
            }
            "swapaxes" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                if matches!(val, Value::StaticArray { .. }) {
                    return Some(crate::ops::static_ndarray_ops::ndarray_swapaxes(
                        &mut self.builder,
                        &val,
                        visited_args,
                    ));
                }
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::swapaxes(&mut self.builder, &val, visited_args)
            }
            "nonzero" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                crate::ops::static_ndarray_ops::np_nonzero(&mut self.builder, &[val])
            }
            "mean" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_reduce(
                    &mut self.builder,
                    "mean",
                    &val,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return Some(out);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_mean(
                    &mut self.builder,
                    &all_args,
                    _visited_kwargs,
                )
            }
            method @ ("var" | "std") => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let ddof_arg_orig = visited_kwargs_orig
                    .get("ddof")
                    .or_else(|| visited_args_orig.get(3));
                let unsupported_dtype = visited_kwargs_orig
                    .get("dtype")
                    .or_else(|| visited_args_orig.get(1));
                if !matches!(unsupported_dtype, None | Some(Value::None)) {
                    panic!(
                        "{}: dtype argument is deferred for native StaticArray stats",
                        method
                    );
                }
                let unsupported_out = visited_kwargs_orig
                    .get("out")
                    .or_else(|| visited_args_orig.get(2));
                if !matches!(unsupported_out, None | Some(Value::None)) {
                    panic!(
                        "{}: out argument is deferred for native StaticArray stats",
                        method
                    );
                }
                let keepdims = super::kwarg_static_truthy(visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_var_std(
                    &mut self.builder,
                    method,
                    &val,
                    axis_arg_orig,
                    ddof_arg_orig,
                    keepdims,
                ) {
                    return Some(out);
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                if method == "var" {
                    crate::ops::static_ndarray_ops::np_var(
                        &mut self.builder,
                        &all_args,
                        _visited_kwargs,
                    )
                } else {
                    crate::ops::static_ndarray_ops::np_std(
                        &mut self.builder,
                        &all_args,
                        _visited_kwargs,
                    )
                }
            }
            "cumsum" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_cumsum(
                    &mut self.builder,
                    &all_args,
                    _visited_kwargs,
                )
            }
            "cumprod" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let mut all_args = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_cumprod(
                    &mut self.builder,
                    &all_args,
                    _visited_kwargs,
                )
            }
            "squeeze" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P4c: native StaticArray dispatch.
                if matches!(val, Value::StaticArray { .. }) {
                    let axis_arg = _visited_kwargs.get("axis").or_else(|| visited_args.first());
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_squeeze(
                        &mut self.builder,
                        &val,
                        axis_arg,
                    ) {
                        return Some(out);
                    }
                }
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                let mut all_args: Vec<Value> = vec![val];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_squeeze(
                    &mut self.builder,
                    &all_args,
                    _visited_kwargs,
                )
            }
            "filter" => {
                let val = self.ctx.get(var).unwrap_or(Value::None);
                // P1 segarr boundary: normalize StaticArray to legacy List view.
                let val =
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &val);
                crate::helpers::array_ops::filter(&mut self.builder, &val, visited_args)
            }
            _ => return None,
        };
        Some(v)
    }
}
