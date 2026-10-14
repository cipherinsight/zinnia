use std::collections::HashMap;

use crate::ast::*;
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

use super::IRGenerator;
use crate::builder::IRBuilder;

mod builtins;
mod array_compat;
pub(crate) use array_compat::astype_without_copy;
mod array_iteration;
mod dynamic_array_compat;
mod linalg_norm;
mod list_methods;
mod ndarray_methods;
mod numpy_derived;
mod numpy_funcs;

/// Write `result` into the existing array `target` (element-wise, through
/// views; shapes must match and a float result cannot go into an integer
/// array). Returns `target`.
pub(super) fn write_into(
    b: &mut crate::builder::IRBuilder,
    result: &Value,
    target: &Value,
) -> Value {
    array_compat::write_output(b, result, target, None)
}

/// The compile-time Boolean keyword argument `name` (`keepdims=`,
/// `strict=`), or `default` when it is absent; a runtime value is rejected.
pub(super) fn kwarg_static_truthy(
    kwargs: &HashMap<String, Value>,
    name: &str,
    default: bool,
) -> bool {
    crate::helpers::static_args::static_bool_arg(kwargs.get(name), default, &format!("`{name}=`"))
}

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
    /// A ufunc's output array passed positionally after its inputs
    /// (`np.add(x, 1, y)`); removed from `args` so handlers see only inputs.
    pub positional_out: Option<&'a Value>,
}

impl IRGenerator {
    pub(crate) fn visit_named_attr(&mut self, n: &ASTNamedAttribute) -> Value {
        // Handle starred args: foo(1, *args, 4) → flatten starred into arg list
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                let inner = crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &inner);
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
        let positional_out = split_positional_ufunc_out(target, member, &mut visited_args);

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
        if target==Some("np") {
            if let Some(value)=array_iteration::dispatch(&mut self.builder,member,&visited_args_orig,&visited_kwargs_orig) {return value;}
            if let Some(value)=dynamic_array_compat::dispatch(&mut self.builder,member,&visited_args_orig,&visited_kwargs_orig) {return value;}
        }
        if target == Some("np") && matches!(member, "sum" | "prod" | "min" | "amin" | "max" | "amax" | "any" | "all") {
            let op = match member { "amin" => "min", "amax" => "max", other => other };
            return array_compat::reduction(&mut self.builder,op,&visited_args_orig,&visited_kwargs_orig);
        }
        if target.is_none() && self.registered_chips.contains_key(member) {
            return self.visit_chip_call(member, &visited_args_orig, &visited_kwargs_orig);
        }
        if let Some(target_name) = target {
            // A method or attribute of a variable: the variable must be
            // bound here and have a known shape (`visit_load` checks the
            // same for a bare name). The methods look the variable up
            // themselves, so a stand-in for arrays of different shapes is
            // rejected even where the path would select one of them.
            if let Some(value) = self.ctx.get(target_name) {
                self.usable_name_value(target_name);
                if let Some(reason) = self.builder.unknown_shape_reason_in(&value) {
                    panic!("Using `{}` is not supported here: {}", target_name, reason);
                }
            }
            if target_name == "copy" && matches!(member, "copy" | "deepcopy") {
                return visited_args_orig.first().cloned().unwrap_or(Value::None);
            }
            // A real array's `.real` / `.conjugate()` is itself and its
            // `.imag` is zeros; a complex array is handled with the ndarray
            // methods (`try_ndarray_method`).
            if let Some(Value::StaticArray { shape, .. }) =
                self.ctx.get(target_name).filter(|v| !is_complex_static_array(v))
            {
                match member {
                    "real" | "conjugate" => return self.ctx.get(target_name).unwrap(),
                    "imag" => {
                        let total: usize = shape.iter().product();
                        let zero = self.builder.ir_constant_float(0.0);
                        let zeros = crate::helpers::static_array::base::build_static_array_from_flat(
                            &mut self.builder,
                            vec![zero; total],
                            shape,
                            crate::types::NumberType::Float,
                        );
                        self.builder.mark_read_only_in(&zeros, READ_ONLY_RESULT);
                        return zeros;
                    }
                    _ => {}
                }
            }
        }
        let visited_args: Vec<Value> = visited_args
            .iter()
            .map(|v| crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, v))
            .collect();
        let _visited_kwargs: HashMap<String, Value> = _visited_kwargs
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &v),
                )
            })
            .collect();

        let ctx = DispatchCtx {
            target,
            member,
            args: &visited_args,
            kwargs: &_visited_kwargs,
            args_orig: &visited_args_orig,
            kwargs_orig: &visited_kwargs_orig,
            ast_node: n,
            positional_out: positional_out.as_ref(),
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
            return self.finish_numpy_call(&ctx, v);
        }
        if let Some(v) = self.try_list_method(&ctx) {
            return v;
        }
        if let Some(v) = self.try_ndarray_method(&ctx) {
            return v;
        }
        if let Some(v) = self.try_np_fallback(&ctx) {
            return self.finish_numpy_call(&ctx, v);
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

    /// The result of an `np.*` call as NumPy returns it: a function that
    /// returns one ndarray returns an ndarray (see [`returns_one_ndarray`]),
    /// and NumPy's `out` argument is honoured: the result is written into
    /// the given array (through views) and that array returned. Ufuncs also
    /// accept `out` positionally, right after their inputs
    /// (`np.add(x, 1, y)`).
    fn finish_numpy_call(&mut self, ctx: &DispatchCtx, value: Value) -> Value {
        if ctx.target != Some("np") {
            return value;
        }
        let value = if returns_one_ndarray(ctx.member, ctx.args_orig.len()) {
            self.numeric_sequence_as_ndarray(value)
        } else {
            value
        };
        // NumPy returns these read-only: a write to them raises. The
        // diagonal of a 2-D array (`np.diag(m)`, `np.diagonal(a)`) is a
        // read-only view; the diagonal matrix built from a 1-D array is a
        // new, writable array.
        let real_argument = ctx.args_orig.first().is_some_and(|a| !is_complex_value(a));
        let diagonal_of_matrix = (ctx.member == "diag" && ctx.args_orig.first().is_some_and(is_matrix))
            || ctx.member == "diagonal";
        if ctx.member == "broadcast_to" || (ctx.member == "imag" && real_argument) || diagonal_of_matrix {
            self.builder.mark_read_only_in(&value, READ_ONLY_RESULT);
        }
        // NumPy answers these with a view of their first argument: where
        // Zinnia's result is a copy, writes through either one are tracked
        // (`detached_copies`).
        if RETURNS_A_VIEW.contains(&ctx.member) || diagonal_of_matrix {
            if let Some(base) = ctx.args_orig.first() {
                self.record_view_result(base, &value);
            }
        }
        if ctx.member == "clip" {
            return value;
        }
        let target = match (ctx.positional_out, ctx.kwargs_orig.get("out")) {
            (Some(_), Some(_)) => panic!("np.{}: out given both positionally and as a keyword", ctx.member),
            (Some(out), None) | (None, Some(out)) => out,
            (None, None) => return value,
        };
        let written = array_compat::write_output(&mut self.builder, &value, target, ctx.kwargs_orig.get("where"));
        self.note_array_write(&written);
        written
    }

    /// A numeric Python sequence (possibly holding arrays) as a
    /// `StaticArray`; any other value as it is. Several `np.*` lowerings
    /// build their result as a nested `List`; returned as is, the result
    /// picks up Python sequence semantics downstream, so that
    /// `2 * np.arctan2(a, b)` or `2 * np.cumsum([x, y])` repeated the list
    /// instead of doubling every element.
    fn numeric_sequence_as_ndarray(&mut self, value: Value) -> Value {
        if !matches!(value, Value::List(_) | Value::Tuple(_)) {
            return value;
        }
        let nested = crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &value);
        let shape = crate::helpers::composite::get_composite_shape(&nested);
        let rectangular = crate::helpers::composite::flatten_composite(&nested).len()
            == shape.iter().product::<usize>();
        if !rectangular {
            return value;
        }
        crate::helpers::static_array::base::to_static_array(&mut self.builder, &nested).unwrap_or(value)
    }

    pub(crate) fn visit_expr_attr(&mut self, n: &ASTExprAttribute) -> Value {
        if let Some(out) = self.try_np_linalg_expr_attr(n) {
            return out;
        }

        let target = self.visit(&n.target);
        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                let inner = crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &inner);
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
        if matches!(n.member.as_str(), "sum" | "prod" | "min" | "max" | "any" | "all") {
            let mut args = vec![target.clone()]; args.extend(visited_args.iter().cloned());
            return array_compat::reduction(&mut self.builder, &n.member, &args, &visited_kwargs);
        }
        if n.member=="squeeze" {
            if let Some(value)=crate::helpers::static_array::shape::try_apply_squeeze(&mut self.builder,&target,visited_kwargs.get("axis").or_else(||visited_args.first())) {return value;}
        }
        if let Some(metadata) = static_array_metadata(&mut self.builder, &n.member, &target) {
            return metadata;
        }
        let target_orig = target.clone();
        let visited_args_orig: Vec<Value> = visited_args.clone();
        let visited_kwargs_orig: HashMap<String, Value> = visited_kwargs.clone();
        let target =
            crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &target);
        let visited_args: Vec<Value> = visited_args
            .iter()
            .map(|v| crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, v))
            .collect();
        let visited_kwargs: HashMap<String, Value> = visited_kwargs
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &v),
                )
            })
            .collect();

        // DynamicNDArray dispatch — route to dedicated handler
        if let Value::DynamicNDArray(data) = &target {
            let data = data.clone();
            let result = self.dispatch_dyn_ndarray_method(
                target,
                n.member.as_str(),
                &visited_args,
                &visited_kwargs,
            );
            self.note_metadata_bounds(&data, n.member.as_str(), &result);
            return result;
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
                    return Value::Complex {
                        real: real.clone(),
                        imag: ni,
                    };
                }
                _ => {}
            }
        }

        if matches!(
            target,
            Value::Integer(_) | Value::Float(_) | Value::Boolean(_)
        ) {
            match n.member.as_str() {
                "real" | "conjugate" => return target,
                "imag" => {
                    return if matches!(target, Value::Float(_)) {
                        self.builder.ir_constant_float(0.0)
                    } else {
                        self.builder.ir_constant_int(0)
                    };
                }
                _ => {}
            }
        }

        if is_complex_static_array(&target_orig) {
            use crate::helpers::static_array::complex::{
                np_conj_static_array, np_imag_static_array, np_real_static_array,
            };
            match n.member.as_str() {
                "real" => return np_real_static_array(&mut self.builder, &target_orig),
                "imag" => return np_imag_static_array(&mut self.builder, &target_orig),
                "conjugate" => return np_conj_static_array(&mut self.builder, &target_orig),
                _ => {}
            }
        }

        // A real array's `.real` / `.conjugate()` is itself, its `.imag` zeros.
        if matches!(target_orig, Value::StaticArray { .. }) {
            match n.member.as_str() {
                "real" | "conjugate" => return target_orig,
                "imag" => {
                    if let Value::StaticArray { shape, .. } = &target_orig {
                        let total: usize = shape.iter().product();
                        let zero = self.builder.ir_constant_float(0.0);
                        let zeros = crate::helpers::static_array::base::build_static_array_from_flat(
                            &mut self.builder,
                            vec![zero; total],
                            shape.clone(),
                            crate::types::NumberType::Float,
                        );
                        self.builder.mark_read_only_in(&zeros, READ_ONLY_RESULT);
                        return zeros;
                    }
                }
                _ => {}
            }
        }

        match n.member.as_str() {
            "clip" => {
                let mut args = vec![target_orig.clone()];
                args.extend(visited_args_orig.iter().cloned());
                array_compat::clip(&mut self.builder, &args, &visited_kwargs_orig)
            }
            "copy" => array_compat::copy_array(&mut self.builder, &target_orig),
            // `sum`/`prod`/`min`/`max`/`any`/`all` returned early through
            // `array_compat::reduction` above.
            "mean" => {
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = kwarg_static_truthy(&visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_reduce(
                    &mut self.builder,
                    "mean",
                    &target_orig,
                    axis_arg_orig,
                    keepdims,
                ) {
                    return out;
                }
                let mut all_args = vec![target.clone()];
                all_args.extend(visited_args.iter().cloned());
                crate::ops::static_ndarray_ops::np_mean(&mut self.builder, &all_args, &visited_kwargs)
            }
            method @ ("argmax" | "argmin") => {
                // P4b: native StaticArray dispatch before the legacy boundary.
                let axis_arg_orig = visited_kwargs_orig
                    .get("axis")
                    .or_else(|| visited_args_orig.first());
                let keepdims = kwarg_static_truthy(&visited_kwargs_orig, "keepdims", false);
                if let Some(out) = crate::helpers::static_array::reductions::try_apply_argmax_argmin(
                    &mut self.builder,
                    &target_orig,
                    axis_arg_orig,
                    method == "argmax",
                    keepdims,
                ) {
                    return out;
                }
                let axis_arg = visited_kwargs.get("axis").or_else(|| visited_args.first());
                crate::helpers::array_ops::argmax_argmin(
                    &mut self.builder,
                    &target,
                    axis_arg,
                    method == "argmax",
                )
            }
            "transpose" => {
                let args = if let Some(axes_val) = visited_kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    visited_args.clone()
                };
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_transpose(
                        &mut self.builder,
                        &target_orig,
                        &args,
                    ) {
                        return out;
                    }
                }
                crate::helpers::array_ops::transpose(&mut self.builder, &target, &args)
            }
            "T" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_transpose(
                        &mut self.builder,
                        &target_orig,
                        &[],
                    ) {
                        return out;
                    }
                }
                crate::helpers::ndarray::ndarray_transpose(&mut self.builder, &target, &[])
            }
            "tolist" => target,
            "astype" => {
                let dtype = visited_args_orig.first().or_else(|| visited_kwargs_orig.get("dtype"))
                    .expect("astype requires a dtype");
                if array_compat::astype_without_copy(&visited_kwargs_orig) {
                    let dtype = crate::helpers::dtype::Dtype::required(dtype, "astype");
                    return self.as_dtype_without_copy(&target_orig, Some(dtype), false, "astype");
                }
                array_compat::astype(&mut self.builder, &target_orig, dtype)
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
            "flatten" | "ravel" | "flat" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    if let Some(out) = crate::helpers::static_array::shape::try_apply_flatten(
                        &mut self.builder,
                        &target_orig,
                        false,
                    ) {
                        return out;
                    }
                }
                let flat = crate::helpers::composite::flatten_composite(&target);
                let types = flat.iter().map(|v| v.zinnia_type()).collect();
                let out = Value::List(CompositeData {
                    elements_type: types,
                    values: flat,
                    value_id: ValueId::next(),
                });
                out
            }
            "reshape" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::reshape(
                        &mut self.builder,
                        &target_orig,
                        &visited_args,
                    );
                }
                crate::helpers::array_ops::reshape(&mut self.builder, &target, &visited_args)
            }
            "moveaxis" => {
                // P4c: native StaticArray dispatch.
                if matches!(target_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::moveaxis(
                        &mut self.builder,
                        &target_orig,
                        &visited_args,
                    );
                }
                crate::helpers::array_ops::moveaxis(&mut self.builder, &target, &visited_args)
            }
            "swapaxes" => {
                if matches!(target_orig, Value::StaticArray { .. }) {
                    return crate::helpers::array_ops::swapaxes(
                        &mut self.builder,
                        &target_orig,
                        &visited_args,
                    );
                }
                crate::helpers::array_ops::swapaxes(&mut self.builder, &target, &visited_args)
            }
            "nonzero" => crate::ops::static_ndarray_ops::np_nonzero(
                &mut self.builder,
                &[target_orig.clone()],
            ),
            "repeat" => crate::helpers::array_ops::repeat(
                &mut self.builder,
                &target,
                &visited_args,
                &visited_kwargs,
            ),
            "filter" => {
                crate::helpers::array_ops::filter(&mut self.builder, &target, &visited_args)
            }
            "shape" => {
                let shape = crate::helpers::composite::get_composite_shape(&target);
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
            "dtype" => crate::helpers::dtype::dtype_of(&mut self.builder, &target_orig),
            "append" | "extend" | "pop" | "remove" | "clear" | "reverse" | "count"
            | "index" => {
                // These should be handled via visit_named_attr with a target variable name
                panic!(
                    "Expr attribute `.{}` on non-variable target not supported",
                    n.member
                )
            }
            _ => {
                panic!(
                    "Expr attribute `.{}` not yet implemented in Rust IR generator",
                    n.member
                )
            }
        }
    }

    fn try_np_linalg_expr_attr(&mut self, n: &ASTExprAttribute) -> Option<Value> {
        let ASTNode::ASTNamedAttribute(target) = &*n.target else {
            return None;
        };
        if target.target.as_deref() != Some("np") || target.member != "linalg" {
            return None;
        }
        if n.member == "norm" {
            return Some(self.visit_np_linalg_norm(n));
        }

        let mut visited_args: Vec<Value> = Vec::new();
        for a in &n.args {
            if let ASTNode::ASTStarredExpr(se) = a {
                let inner = self.visit(&se.value);
                let inner = crate::helpers::static_array::base::deep_to_value_list(&mut self.builder, &inner);
                if let Value::List(data) | Value::Tuple(data) = inner {
                    visited_args.extend(data.values);
                } else {
                    visited_args.push(inner);
                }
            } else {
                visited_args.push(self.visit(a));
            }
        }
        if !n.kwargs.is_empty() {
            panic!("np.linalg.{}: keyword arguments are not supported", n.member);
        }
        if visited_args
            .iter()
            .any(|arg| matches!(arg, Value::DynamicNDArray(_)))
        {
            panic!(
                "np.linalg.{} is not yet supported for bounded-dynamic arrays; use static arrays or keep linalg outside the bounded-dynamic surface",
                n.member
            );
        }

        let out = match n.member.as_str() {
            "inv" | "solve" | "cholesky" => {
                crate::ops::static_ndarray_ops::np_linalg_implicit_result(n.member.as_str())
            }
            "det" => {
                let [a] = visited_args.as_slice() else {
                    panic!("np.linalg.det takes one argument, got {}", visited_args.len());
                };
                crate::ops::static_ndarray_ops::np_linalg_det(&mut self.builder, a)
            }
            "verify_inverse" => crate::ops::static_ndarray_ops::np_linalg_verify_inverse(
                &mut self.builder,
                &visited_args,
            ),
            "verify_solve" => crate::ops::static_ndarray_ops::np_linalg_verify_solve(
                &mut self.builder,
                &visited_args,
            ),
            "verify_cholesky" => crate::ops::static_ndarray_ops::np_linalg_verify_cholesky(
                &mut self.builder,
                &visited_args,
            ),
            other => panic!(
                "np.linalg.{} is not implemented; supported: norm, det and the verification helpers verify_inverse, verify_solve, verify_cholesky",
                other
            ),
        };
        Some(out)
    }
}

fn is_complex_static_array(value: &Value) -> bool {
    matches!(value, Value::StaticArray { dtype: crate::types::NumberType::Complex, .. })
}

/// Whether `np.<name>` called with `positional` arguments returns exactly
/// one ndarray in NumPy (when its result is not a scalar): every ufunc, and
/// the array-building and array-reshaping functions below. Functions that
/// return a tuple or list of arrays (`np.nonzero`, `np.split`,
/// `np.histogram`, `np.unique` with `return_counts`, one-argument
/// `np.where`, several-argument `np.atleast_*d`, ...) are not listed.
fn returns_one_ndarray(name: &str, positional: usize) -> bool {
    const ONE_NDARRAY: &[&str] = &[
        "add_outer", "subtract_outer", "multiply_outer", "divide_outer", "arange", "argwhere",
        "bincount", "block", "broadcast_to", "ceil", "clip", "column_stack", "concatenate", "convolve",
        "copy", "corrcoef", "correlate", "cov", "cumprod", "cumsum", "delete", "diag", "diff",
        "dot", "dstack", "einsum", "empty_like", "expand_dims", "flatnonzero", "flip", "fliplr",
        "flipud", "floor", "full", "full_like", "hstack", "in1d", "inner", "insert", "isclose", "isin",
        "linspace", "matmul", "mean", "median", "moveaxis", "ones", "ones_like", "outer", "pad",
        "percentile", "reciprocal", "repeat", "reshape", "rot90", "round", "around", "row_stack", "select",
        "square", "squeeze", "stack", "swapaxes", "take", "take_along_axis", "tensordot", "tile",
        "transpose", "tril", "triu", "trunc", "vstack", "zeros", "zeros_like",
    ];
    match name {
        "where" => positional == 3,
        "atleast_1d" | "atleast_2d" | "atleast_3d" => positional == 1,
        _ => ONE_NDARRAY.contains(&name) || ufunc_input_count(name).is_some(),
    }
}

/// Remove and return the positional `out` argument of a NumPy ufunc call
/// (`np.add(x, 1, y)`), so the handler sees only the inputs.
fn split_positional_ufunc_out(target: Option<&str>, member: &str, args: &mut Vec<Value>) -> Option<Value> {
    if target != Some("np") {
        return None;
    }
    let inputs = ufunc_input_count(member)?;
    if args.len() > inputs + 1 {
        panic!(
            "np.{}: takes {} input(s) and one output, got {} positional arguments",
            member,
            inputs,
            args.len()
        );
    }
    (args.len() == inputs + 1).then(|| args.pop().expect("length checked"))
}

/// Number of inputs of a NumPy ufunc called as `np.<name>`, or `None` when
/// `name` is not a single-output ufunc. A ufunc takes its `out` array as the
/// positional argument after its inputs.
fn ufunc_input_count(name: &str) -> Option<usize> {
    const BINARY: &[&str] = &[
        "add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "power",
        "float_power", "mod", "remainder", "fmod", "maximum", "minimum", "fmax", "fmin",
        "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor",
        "left_shift", "right_shift", "greater", "greater_equal", "less", "less_equal", "equal",
        "not_equal", "arctan2", "hypot", "copysign", "ldexp", "logaddexp", "logaddexp2",
        "heaviside", "gcd", "lcm", "nextafter", "matmul",
    ];
    const UNARY: &[&str] = &[
        "negative", "positive", "absolute", "abs", "fabs", "sign", "sqrt", "cbrt", "square",
        "reciprocal", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos",
        "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh",
        "arctanh", "deg2rad", "rad2deg", "degrees", "radians", "floor", "ceil", "trunc", "rint",
        "conjugate", "conj", "logical_not", "invert", "bitwise_not", "isnan", "isinf",
        "isfinite", "signbit", "spacing",
    ];
    if BINARY.contains(&name) {
        Some(2)
    } else if UNARY.contains(&name) {
        Some(1)
    } else {
        None
    }
}

/// `np.*` functions NumPy answers with a view of their first argument (when
/// it is an array): a write through the result changes the argument, and
/// the other way round.
const RETURNS_A_VIEW: &[&str] = &[
    "atleast_1d", "atleast_2d", "atleast_3d", "broadcast_to", "expand_dims", "flip", "fliplr",
    "flipud", "moveaxis", "rot90", "squeeze", "swapaxes", "transpose",
];

/// Why a result NumPy returns read-only cannot be written.
const READ_ONLY_RESULT: &str = "NumPy returns this array read-only (`np.broadcast_to`, \
     `np.diag` of a 2-D array, or the `imag` of a real array) and raises `ValueError` on a write \
     to it; write to a copy instead";

/// `.shape`, `.ndim` or `.size` of a static array, from its layout: the
/// cells are not read, so the metadata of an array whose cells are stale
/// (they miss a write NumPy would show, `builder/stale.rs`) is still
/// available. `None` for another member or value.
fn static_array_metadata(b: &mut IRBuilder, member: &str, v: &Value) -> Option<Value> {
    let Value::StaticArray { shape, .. } = v else { return None };
    match member {
        "shape" => {
            let extents: Vec<Value> = shape
                .iter()
                .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                .collect();
            Some(Value::Tuple(CompositeData {
                elements_type: vec![ZinniaType::Integer; extents.len()],
                values: extents,
                value_id: ValueId::next(),
            }))
        }
        "ndim" => Some(b.ir_constant_int(shape.len() as i64)),
        "size" => Some(b.ir_constant_int(shape.iter().product::<usize>() as i64)),
        _ => None,
    }
}

/// Whether `v` is a 2-D array (`np.diag` of one is its diagonal).
fn is_matrix(v: &Value) -> bool {
    match v {
        Value::StaticArray { shape, .. } => shape.len() == 2,
        Value::DynamicNDArray(data) => data.envelope.dims.len() == 2,
        Value::List(data) | Value::Tuple(data) => {
            data.values.first().is_some_and(|row| matches!(row, Value::List(_) | Value::Tuple(_)))
        }
        _ => false,
    }
}

/// Whether `v` is a complex number or array.
fn is_complex_value(v: &Value) -> bool {
    match v {
        Value::Complex { .. } => true,
        Value::StaticArray { dtype, .. } => *dtype == crate::types::NumberType::Complex,
        Value::DynamicNDArray(data) => data.dtype == crate::types::NumberType::Complex,
        Value::List(d) | Value::Tuple(d) => d.values.iter().any(is_complex_value),
        _ => false,
    }
}
