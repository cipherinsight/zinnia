use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

use super::IRGenerator;

mod builtins;
mod numpy_funcs;
mod list_methods;
mod ndarray_methods;

/// Bundle of dispatcher inputs shared across category sub-dispatchers.
///
/// Each `try_*` helper receives a borrowed `DispatchCtx` so that it can
/// inspect the same args / kwargs (both the legacy `Value::List`-normalised
/// view and the un-converted `_orig` view used by the segment-backed
/// `StaticArray` fast-paths) without having to be threaded a long argument
/// list. The `ast_node` field is the original `ASTNamedAttribute`, needed by
/// the `np.block` arm to walk the leftmost AST path.
pub(crate) struct DispatchCtx<'a> {
    pub target: Option<&'a str>,
    pub member: &'a str,
    pub args: &'a [Value],
    pub kwargs: &'a HashMap<String, Value>,
    pub args_orig: &'a [Value],
    pub kwargs_orig: &'a HashMap<String, Value>,
    pub ast_node: &'a ASTNamedAttribute,
}

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

        let ctx = DispatchCtx {
            target,
            member,
            args: &visited_args,
            kwargs: &_visited_kwargs,
            args_orig: &visited_args_orig,
            kwargs_orig: &visited_kwargs_orig,
            ast_node: n,
        };

        // Category dispatch chain. Order is preserved from the pre-split
        // big match: builtins (target=None) → numpy funcs (target="np") →
        // DynamicNDArray class methods → list methods → complex/StaticArray
        // accessors → DynamicNDArray method dispatch → ndarray methods /
        // property access → np.* registry fallback → chip / external calls.
        if let Some(v) = self.try_builtin(&ctx) {
            return v;
        }
        if let Some(v) = self.try_numpy_func(&ctx) {
            return v;
        }
        if let Some(v) = self.try_list_method(&ctx) {
            return v;
        }
        if let Some(v) = self.try_ndarray_method(&ctx) {
            return v;
        }
        if let Some(v) = self.try_np_fallback(&ctx) {
            return v;
        }

        // ── Chip calls (no target or target not a known variable) ─
        if target.is_none() && self.registered_chips.contains_key(member) {
            return self.visit_chip_call(member, &visited_args, &_visited_kwargs);
        }

        // ── External function calls ─────────────────────────────────
        if target.is_none() && self.registered_externals.contains_key(member) {
            return self.visit_external_call(member, &visited_args);
        }

        // ── Fallback ───────────────────────────────────────────────
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
                let out = Value::List(CompositeData { elements_type: types, values: flat, value_id: ValueId::next() });
                if let (Some(in_vid), Some(out_vid)) = (target_orig.value_id(), out.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(&mut self.builder, in_vid, out_vid);
                }
                out
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
                Value::Tuple(CompositeData { elements_type: types, values: shape_vals, value_id: ValueId::next() })
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
