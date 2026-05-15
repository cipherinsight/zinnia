use crate::builder::IRBuilder;
use crate::types::{CompositeData, SliceIndex, Value, ValueId};
use super::composite;

pub fn ndarray_transpose(_b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    // Determine the shape of the input
    let shape = composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim <= 1 { return val.clone(); }

    // Determine axis permutation — check length before validating individual values
    let raw_axes: Vec<i64> = if args.is_empty() || matches!(args.first(), Some(Value::None)) {
        (0..ndim as i64).rev().collect()
    } else if let Some(Value::Tuple(perm_data)) | Some(Value::List(perm_data)) = args.first() {
        perm_data.values.iter().map(|v| v.int_val().unwrap_or(0)).collect()
    } else {
        args.iter().map(|v| v.int_val().unwrap_or(0)).collect()
    };

    // Check length first (before resolving individual values)
    if raw_axes.len() != ndim {
        panic!("Length of `axes` should be equal to the number of dimensions of the array (expected {}, got {})", ndim, raw_axes.len());
    }

    let axes: Vec<usize> = raw_axes.iter().map(|&a| {
        let resolved = if a < 0 { ndim as i64 + a } else { a };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
        }
        resolved as usize
    }).collect();
    // Check for invalid axis values
    for &a in &axes {
        if a >= ndim {
            panic!("Invalid axis value: {} is out of bounds for array with {} dimensions", a, ndim);
        }
    }
    // Check for valid permutation (no duplicates)
    let mut seen = vec![false; ndim];
    for &a in &axes {
        if seen[a] {
            panic!("axes should be a permutation of 0 to {}", ndim - 1);
        }
        seen[a] = true;
    }

    // Calculate output shape
    let out_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();

    // Flatten the input, then reassemble in transposed order
    let flat = composite::flatten_composite(val);
    if flat.is_empty() { return val.clone(); }

    // Compute strides for input
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    // Compute strides for output
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let total: usize = shape.iter().product();
    let mut out_flat = vec![Value::None; total];

    // For each element in the flat array, compute its input index tuple,
    // permute it, and write to the output position
    for flat_idx in 0..total {
        // Compute input multi-index
        let mut remainder = flat_idx;
        let mut in_idx = vec![0usize; ndim];
        for d in 0..ndim {
            in_idx[d] = remainder / in_strides[d];
            remainder %= in_strides[d];
        }
        // Permute to get output multi-index
        let mut out_idx = vec![0usize; ndim];
        for d in 0..ndim {
            out_idx[d] = in_idx[axes[d]];
        }
        // Compute output flat index
        let mut out_flat_idx = 0;
        for d in 0..ndim {
            out_flat_idx += out_idx[d] * out_strides[d];
        }
        out_flat[out_flat_idx] = flat[flat_idx].clone();
    }

    // Rebuild nested structure from output shape
    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    composite::build_nested_value(out_flat, types, &out_shape)
}

pub fn ndarray_argmax_argmin(b: &mut IRBuilder, val: &Value, _args: &[Value], is_max: bool) -> Value {
    let elements = composite::flatten_composite(val);
    if elements.is_empty() { return b.ir_constant_int(0); }
    let len_arr = elements.len();
    let mut best_idx = b.ir_constant_int(0);
    let mut best_val = elements[0].clone();
    for (i, elem) in elements.iter().enumerate().skip(1) {
        let cond = if is_max {
            b.ir_greater_than_i(elem, &best_val)
        } else {
            b.ir_less_than_i(elem, &best_val)
        };
        let idx_val = b.ir_constant_int(i as i64);
        best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
        best_val = b.ir_select_i(&cond, elem, &best_val);
    }
    if let Some(idx_vid) = best_idx.value_id() {
        let len_arr_val = b.ir_constant_int(len_arr as i64);
        if let Some(len_arr_vid) = len_arr_val.value_id() {
            let mut formals = std::collections::HashMap::new();
            formals.insert("len_arr".to_string(), len_arr_vid);
            b.fire_contract("dyn_argextremum", idx_vid, &formals);
        }
    }
    best_idx
}

/// True if every leaf of `val` is a compile-time-constant boolean.
fn all_const_bool(val: &Value) -> bool {
    match val {
        Value::List(d) | Value::Tuple(d) => !d.values.is_empty() && d.values.iter().all(all_const_bool),
        Value::Boolean(_) => val.bool_val().is_some(),
        _ => false,
    }
}

/// True if every leaf of `val` is a compile-time-constant integer
/// (booleans excluded — those should be routed to boolean masking).
fn all_const_int_strict(val: &Value) -> bool {
    match val {
        Value::List(d) | Value::Tuple(d) => !d.values.is_empty() && d.values.iter().all(all_const_int_strict),
        Value::Integer(_) => val.int_val().is_some(),
        _ => false,
    }
}

/// Static boolean masking: `data[mask]` where `mask` is a static-shape ndarray
/// of compile-time-known booleans. Same-shape only — prefix masking is left
/// for the future bounded-dynamic envelope work.
///
/// Returns Err with a descriptive message on shape mismatch; the caller is
/// expected to forward that to the user as a hard error.
pub fn boolean_mask_static(data: &Value, mask: &Value) -> Result<Value, String> {
    let dshape = super::composite::get_composite_shape(data);
    let mshape = super::composite::get_composite_shape(mask);
    if dshape != mshape {
        return Err(format!(
            "boolean mask shape {:?} must match array shape {:?} (prefix masks not yet supported)",
            mshape, dshape
        ));
    }
    let data_flat = super::composite::flatten_composite(data);
    let mask_flat = super::composite::flatten_composite(mask);
    let mut selected: Vec<Value> = Vec::new();
    for (d, m) in data_flat.into_iter().zip(mask_flat.iter()) {
        if m.bool_val() == Some(true) {
            selected.push(d);
        }
    }
    let types = selected.iter().map(|v| v.zinnia_type()).collect();
    Ok(Value::List(CompositeData { elements_type: types, values: selected, value_id: ValueId::next() }))
}

/// Static fancy indexing along axis 0: `data[idx_array]` where `idx_array` is
/// a static-shape ndarray of compile-time-known integers. The result has
/// shape `idx_array.shape + data.shape[1:]` (NumPy semantics).
pub fn fancy_index_static(data: &CompositeData, idx_array: &Value) -> Result<Value, String> {
    fn walk(data: &CompositeData, idx: &Value) -> Result<Value, String> {
        match idx {
            Value::List(d) | Value::Tuple(d) => {
                let mut out = Vec::with_capacity(d.values.len());
                for v in &d.values {
                    out.push(walk(data, v)?);
                }
                let types = out.iter().map(|v| v.zinnia_type()).collect();
                Ok(Value::List(CompositeData { elements_type: types, values: out, value_id: ValueId::next() }))
            }
            _ => {
                let i = idx.int_val().ok_or_else(|| {
                    "fancy index value is not compile-time constant".to_string()
                })?;
                let len = data.values.len() as i64;
                let i = if i < 0 { len + i } else { i };
                if i < 0 || i >= len {
                    return Err(format!(
                        "index {} out of bounds for axis 0 with size {}",
                        idx.int_val().unwrap(),
                        data.values.len()
                    ));
                }
                Ok(data.values[i as usize].clone())
            }
        }
    }
    walk(data, idx_array)
}

/// Try to dispatch a single-axis composite index as boolean masking or fancy
/// indexing. Returns:
/// - `Ok(Some(value))` on success
/// - `Ok(None)` if the index doesn't look like an advanced index at all
///   (e.g. a heterogeneous list — caller should keep its existing handling)
/// - `Err(msg)` if the index *looks* like advanced indexing but isn't
///   compile-time resolvable; caller should hard-error with `msg`.
pub fn try_advanced_index_static(
    data: &CompositeData,
    idx: &Value,
) -> Result<Option<Value>, String> {
    match idx {
        Value::List(_) | Value::Tuple(_) => {}
        _ => return Ok(None),
    }
    if all_const_bool(idx) {
        return boolean_mask_static(&Value::List(data.clone()), idx).map(Some);
    }
    if all_const_int_strict(idx) {
        return fancy_index_static(data, idx).map(Some);
    }
    // Looks like an array-valued index but its leaves aren't compile-time
    // resolvable as either bools or ints. This is the case the user said to
    // hard-error on, with a "to be implemented" hint.
    Err(
        "array-valued indices (boolean masking / fancy indexing) currently \
         require compile-time-constant index values. Non-constant cases \
         (to be implemented: lowering to dynamic ndarray)."
            .into(),
    )
}

/// Apply trailing `np.newaxis` markers to a scalar value, wrapping it in
/// nested length-1 lists. Any non-NewAxis index here is a usage error.
fn apply_trailing_to_scalar(scalar: Value, indices: &[SliceIndex]) -> Value {
    let mut current = scalar;
    for idx in indices.iter().rev() {
        match idx {
            SliceIndex::NewAxis => {
                let t = current.zinnia_type();
                current = Value::List(CompositeData {
                    elements_type: vec![t],
                    values: vec![current],
                
                    value_id: ValueId::next(),
                });
            }
            SliceIndex::Ellipsis => {
                // `...` at a scalar dim is a no-op (zero remaining axes).
            }
            _ => {
                panic!("too many indices for array: subscript indexes a scalar value");
            }
        }
    }
    current
}

pub fn multidim_subscript(b: &mut IRBuilder, data: &CompositeData, indices: &[SliceIndex]) -> Value {
    if indices.is_empty() {
        return Value::List(data.clone());
    }

    // Expand a single Ellipsis (`...`) into the right number of full-range
    // slices, based on the source rank.
    if indices.iter().any(|i| matches!(i, SliceIndex::Ellipsis)) {
        let shape = super::composite::get_composite_shape(&Value::List(data.clone()));
        let consumed: usize = indices
            .iter()
            .filter(|i| matches!(i, SliceIndex::Single(_) | SliceIndex::Range(_, _, _)))
            .count();
        let num_colons = shape.len().saturating_sub(consumed);
        let mut expanded: Vec<SliceIndex> = Vec::with_capacity(indices.len() - 1 + num_colons);
        let mut seen_ellipsis = false;
        for idx in indices {
            match idx {
                SliceIndex::Ellipsis if !seen_ellipsis => {
                    seen_ellipsis = true;
                    for _ in 0..num_colons {
                        expanded.push(SliceIndex::Range(None, None, None));
                    }
                }
                SliceIndex::Ellipsis => {
                    panic!("an index can only have a single ellipsis ('...')");
                }
                other => expanded.push(other.clone()),
            }
        }
        return multidim_subscript(b, data, &expanded);
    }

    // `np.newaxis` / `None`: insert a unit-length axis here without consuming
    // a source dimension. The remaining indices apply to the same `data`.
    if matches!(&indices[0], SliceIndex::NewAxis) {
        let inner = multidim_subscript(b, data, &indices[1..]);
        return Value::List(CompositeData {
            elements_type: vec![inner.zinnia_type()],
            values: vec![inner],
        
            value_id: ValueId::next(),
        });
    }

    match &indices[0] {
        SliceIndex::Single(idx_value) => {
            if let Some(idx) = idx_value.int_val() {
                let i = if idx < 0 { (data.values.len() as i64 + idx) as usize } else { idx as usize };
                if i >= data.values.len() {
                    return Value::None;
                }
                if indices.len() == 1 {
                    return data.values[i].clone();
                }
                // Recurse into the selected element
                match &data.values[i] {
                    Value::List(inner) | Value::Tuple(inner) => {
                        multidim_subscript(b, inner, &indices[1..])
                    }
                    _ => apply_trailing_to_scalar(data.values[i].clone(), &indices[1..]),
                }
            } else {
                // Dynamic index
                if indices.len() == 1 {
                    return crate::helpers::value_ops::dynamic_list_subscript(b, data, idx_value);
                }
                // Dynamic index with further dimensions: apply the remaining
                // indices to each possible row, then mux on the dynamic index.
                // We can't use `dynamic_list_subscript` here because the
                // per-row results may themselves be composites (e.g. when a
                // remaining index is a Range), and `dynamic_list_subscript`
                // uses scalar `ir_select_i` which doesn't traverse lists.
                // Use `select_value`, which recurses through composites.
                let n = data.values.len();
                if n == 0 {
                    return Value::None;
                }
                let mut per_row_results: Vec<Value> = Vec::with_capacity(n);
                for elem in &data.values {
                    if let Value::List(inner) | Value::Tuple(inner) = elem {
                        per_row_results.push(multidim_subscript(b, inner, &indices[1..]));
                    } else {
                        per_row_results.push(elem.clone());
                    }
                }
                let mut acc = per_row_results.last().unwrap().clone();
                for i in (0..n - 1).rev() {
                    let const_i = b.ir_constant_int(i as i64);
                    let cmp = b.ir_equal_i(idx_value, &const_i);
                    acc = crate::helpers::value_ops::select_value(b, &cmp, &per_row_results[i], &acc);
                }
                acc
            }
        }
        SliceIndex::Range(start, stop, step) => {
            let len = data.values.len() as i64;
            let s = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
            let e = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
            let st = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
            let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
            let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;

            let mut selected = Vec::new();
            let mut i = s;
            while (st > 0 && i < e) || (st < 0 && i > e) {
                if i < data.values.len() {
                    if indices.len() == 1 {
                        selected.push(data.values[i].clone());
                    } else {
                        // Apply remaining indices to each selected element
                        match &data.values[i] {
                            Value::List(inner) | Value::Tuple(inner) => {
                                selected.push(multidim_subscript(b, inner, &indices[1..]));
                            }
                            _ => selected.push(apply_trailing_to_scalar(data.values[i].clone(), &indices[1..])),
                        }
                    }
                }
                i = (i as i64 + st) as usize;
            }
            let types = selected.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: selected, value_id: ValueId::next() })
        }
        // Ellipsis and NewAxis are handled above (at function entry); reaching
        // them here means a logic bug.
        SliceIndex::Ellipsis | SliceIndex::NewAxis => unreachable!(
            "Ellipsis / NewAxis should have been handled before the main match"
        ),
    }
}

/// Per-call inputs for the sum/prod strategy sets used by `builtin_reduce`.
/// Carries the flattened element list, the input array's `value_id` (for
/// precondition construction), the element count, a float-flag for typed
/// constant lowering, and the optional per-element ValueIds list used by
/// the post-sweep interval relay. `Value` is `Clone`, so this struct can
/// live behind the framework's `fn(&mut IRBuilder, &Inputs) -> Output`
/// signature.
pub(crate) struct ReductionInputs {
    pub elements: Vec<Value>,
    pub arr_vid: crate::types::ValueId,
    pub n: i64,
    pub any_float: bool,
    pub element_vids: Option<Vec<crate::types::ValueId>>,
}

fn lower_sum_generic_sweep(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    let mut acc = inputs.elements[0].clone();
    for elem in &inputs.elements[1..] {
        acc = crate::helpers::value_ops::apply_binary_op(b, "add", &acc, elem);
    }
    // Per-element interval relay: Output ∈ [N*lo, N*hi].
    if let (Some(vids), Some(out_vid)) = (inputs.element_vids.as_ref(), acc.value_id()) {
        crate::optim::resolver::relay_reduction_output_interval_int(
            b, vids, out_vid, inputs.n,
        );
    }
    acc
}

/// Strategy A for `sum`: `forall_eq_const(arr, 0)` ⇒ output is constant 0.
/// Sound because every element equals 0, so `0 + 0 + ... + 0 == 0`.
fn lower_sum_constant_zero(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    if inputs.any_float {
        b.ir_constant_float(0.0)
    } else {
        b.ir_constant_int(0)
    }
}

/// Strategy B for `sum`: `forall_eq_const(arr, 1)` ⇒ output is constant N.
/// Sound because every element equals 1, so the sum is N (the element
/// count). Guarded by `builtin_reduce`'s empty-input short-circuit, so
/// N >= 1 here.
fn lower_sum_length_times_one(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    if inputs.any_float {
        b.ir_constant_float(inputs.n as f64)
    } else {
        b.ir_constant_int(inputs.n)
    }
}

fn lower_prod_generic_sweep(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    let mut acc = inputs.elements[0].clone();
    for elem in &inputs.elements[1..] {
        acc = crate::helpers::value_ops::apply_binary_op(b, "mul", &acc, elem);
    }
    acc
}

/// Strategy A for `prod`: `forall_eq_const(arr, 0)` ⇒ output is constant 0.
/// Sound because `0 * x == 0` for any `x` (and N >= 1 here).
fn lower_prod_constant_zero(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    if inputs.any_float {
        b.ir_constant_float(0.0)
    } else {
        b.ir_constant_int(0)
    }
}

/// Strategy B for `prod`: `forall_eq_const(arr, 1)` ⇒ output is constant 1.
/// Sound because `1 * 1 * ... * 1 == 1` for any N >= 1.
fn lower_prod_constant_one(b: &mut IRBuilder, inputs: &ReductionInputs) -> Value {
    if inputs.any_float {
        b.ir_constant_float(1.0)
    } else {
        b.ir_constant_int(1)
    }
}

/// Build the `OpStrategySet` for sum/prod gated on `forall_eq_const(arr, k)`
/// for k ∈ {0, 1}. The op author embeds the concrete `arr_vid` in each
/// precondition; the dispatcher walks declared order and short-circuits
/// on the first `Proved` outcome.
fn reduction_strategy_set(
    arr_vid: crate::types::ValueId,
    op: &str,
) -> crate::optim::OpStrategySet<ReductionInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let pred_eq_k = |k: i64| ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(arr_vid)),
            ContractTerm::LitInt(k),
        ],
    };

    match op {
        "sum" => OpStrategySet {
            strategies: vec![
                OpStrategy {
                    name: "forall_eq_const_zero",
                    precondition: pred_eq_k(0),
                    cost_hint: CostHint::O1,
                    lower: lower_sum_constant_zero,
                },
                OpStrategy {
                    name: "forall_eq_const_one",
                    precondition: pred_eq_k(1),
                    cost_hint: CostHint::O1,
                    lower: lower_sum_length_times_one,
                },
            ],
            default: lower_sum_generic_sweep,
        },
        "prod" => OpStrategySet {
            strategies: vec![
                OpStrategy {
                    name: "forall_eq_const_zero",
                    precondition: pred_eq_k(0),
                    cost_hint: CostHint::O1,
                    lower: lower_prod_constant_zero,
                },
                OpStrategy {
                    name: "forall_eq_const_one",
                    precondition: pred_eq_k(1),
                    cost_hint: CostHint::O1,
                    lower: lower_prod_constant_one,
                },
            ],
            default: lower_prod_generic_sweep,
        },
        _ => unreachable!("reduction_strategy_set called with unsupported op `{}`", op),
    }
}

pub fn builtin_reduce(b: &mut IRBuilder, op: &str, val: &Value) -> Value {
    let elements = composite::flatten_composite(val);
    if elements.is_empty() {
        return match op {
            "sum" => b.ir_constant_int(0),
            "any" => b.ir_constant_bool(false),
            "all" => b.ir_constant_bool(true),
            "prod" => b.ir_constant_int(1),
            "min" | "max" => Value::None,
            _ => Value::None,
        };
    }
    // Are any of the leaves floats? If so, the accumulator must use the
    // float ops; otherwise we'd silently feed floats into ir_add_i etc.
    // and corrupt the IR. We route everything through `apply_binary_op`
    // which already handles int/float promotion correctly.
    let any_float = elements.iter().any(|v| matches!(v, Value::Float(_)));
    // Collect element ValueIds for the input-fact-relay below. We
    // gather them up-front because the reduction loop overwrites `acc`
    // in place. An element without a `value_id` (constant scalar) makes
    // the whole list ineligible for the relay — bail out by leaving the
    // slot empty and skipping the relay later.
    let element_vids: Option<Vec<crate::types::ValueId>> = elements
        .iter()
        .map(|v| v.value_id())
        .collect::<Option<Vec<_>>>();
    let n_elements = elements.len() as i64;
    match op {
        "sum" | "prod" => {
            let inputs = ReductionInputs {
                elements: elements.clone(),
                arr_vid: val.value_id().unwrap_or_else(ValueId::next),
                n: n_elements,
                any_float,
                element_vids: element_vids.clone(),
            };
            let op_name: &'static str = if op == "sum" { "sum" } else { "prod" };
            match val.value_id() {
                Some(_) => {
                    let set = reduction_strategy_set(inputs.arr_vid, op_name);
                    crate::optim::dispatch_strategy(b, op_name, &inputs, &set)
                }
                None => {
                    if op_name == "sum" {
                        lower_sum_generic_sweep(b, &inputs)
                    } else {
                        lower_prod_generic_sweep(b, &inputs)
                    }
                }
            }
        }
        "any" => {
            let mut acc = crate::helpers::value_ops::to_scalar_bool(b, &elements[0]);
            for elem in &elements[1..] {
                let bool_val = crate::helpers::value_ops::to_scalar_bool(b, elem);
                acc = b.ir_logical_or(&acc, &bool_val);
            }
            if let Some(vid) = acc.value_id() {
                b.fire_contract("any", vid, &std::collections::HashMap::new());
            }
            acc
        }
        "all" => {
            let mut acc = crate::helpers::value_ops::to_scalar_bool(b, &elements[0]);
            for elem in &elements[1..] {
                let bool_val = crate::helpers::value_ops::to_scalar_bool(b, elem);
                acc = b.ir_logical_and(&acc, &bool_val);
            }
            if let Some(vid) = acc.value_id() {
                b.fire_contract("all", vid, &std::collections::HashMap::new());
            }
            acc
        }
        "min" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                let cond = crate::helpers::value_ops::apply_binary_op(b, "lt", &acc, elem);
                acc = crate::helpers::value_ops::select_value(b, &cond, &acc, elem);
            }
            // Per-element interval relay: Output ∈ [lo, hi] (multiplier 1).
            if let (Some(vids), Some(out_vid)) = (element_vids.as_ref(), acc.value_id()) {
                crate::optim::resolver::relay_reduction_output_interval_int(
                    b, vids, out_vid, 1,
                );
            }
            acc
        }
        "max" => {
            let mut acc = elements[0].clone();
            for elem in &elements[1..] {
                let cond = crate::helpers::value_ops::apply_binary_op(b, "gt", &acc, elem);
                acc = crate::helpers::value_ops::select_value(b, &cond, &acc, elem);
            }
            // Per-element interval relay: Output ∈ [lo, hi] (multiplier 1).
            if let (Some(vids), Some(out_vid)) = (element_vids.as_ref(), acc.value_id()) {
                crate::optim::resolver::relay_reduction_output_interval_int(
                    b, vids, out_vid, 1,
                );
            }
            acc
        }
        _ => Value::None,
    }
}
