//! Element-wise composite ops: the bridge helpers
//! (`elementwise_binary`, `elementwise_minmax`), the unary numeric ops
//! (`np_floor`, `np_ceil`, `np_round`, `np_trunc`, `np_reciprocal`,
//! `np_square`), `np_diff`, `np_outer`, `np_allclose`, `np_where`,
//! `np_clip`, plus the Python-builtin helpers `range`/`len`/`enumerate`
//! that share the same composite-Value plumbing.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

/// Element-wise binary operation on two composites (bypasses list concatenation).
pub fn elementwise_binary(b: &mut IRBuilder, op: &str, a: &Value, bv: &Value) -> Value {
    match (a, bv) {
        (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
        | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
            if ad.values.len() == bd.values.len() => {
            let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                .map(|(x, y)| elementwise_binary(b, op, x, y))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() })
        }
        _ => crate::helpers::value_ops::apply_scalar_binary_op(b, op, a, bv),
    }
}

/// Element-wise min or max of two composites.
pub fn elementwise_minmax(b: &mut IRBuilder, a: &Value, bv: &Value, is_max: bool) -> Value {
    match (a, bv) {
        (Value::List(ad), Value::List(bd)) | (Value::Tuple(ad), Value::List(bd))
        | (Value::List(ad), Value::Tuple(bd)) | (Value::Tuple(ad), Value::Tuple(bd))
            if ad.values.len() == bd.values.len() => {
            let results: Vec<Value> = ad.values.iter().zip(bd.values.iter())
                .map(|(x, y)| elementwise_minmax(b, x, y, is_max))
                .collect();
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() })
        }
        _ => {
            let cond = if is_max {
                b.ir_greater_than_i(a, bv)
            } else {
                b.ir_less_than_i(a, bv)
            };
            b.ir_select_i(&cond, a, bv)
        }
    }
}

/// Element-wise square: x * x
pub fn np_square(b: &mut IRBuilder, args: &[Value]) -> Value {
    let x = match args.first() {
        Some(v) => v,
        None => return Value::None,
    };
    elementwise_binary(b, "mul", x, x)
}

/// Discrete difference along the last axis: `out[i] = x[i+1] - x[i]` over 1-D
/// composites, recursively applied to inner axes for higher-rank arrays.
pub fn np_diff(b: &mut IRBuilder, args: &[Value]) -> Value {
    let x = match args.first() {
        Some(v) => v,
        None => return Value::None,
    };
    let n = match args.get(1).and_then(|v| v.int_val()) {
        Some(n) => n.max(0) as usize,
        None => 1,
    };
    let mut current = x.clone();
    for _ in 0..n {
        current = diff_once(b, &current);
    }
    current
}

fn diff_once(b: &mut IRBuilder, x: &Value) -> Value {
    match x {
        Value::List(data) | Value::Tuple(data) => {
            // If inner elements are themselves composites, recurse on each row;
            // numpy's np.diff defaults to the last axis.
            if let Some(first) = data.values.first() {
                if matches!(first, Value::List(_) | Value::Tuple(_)) {
                    let rows: Vec<Value> = data.values.iter().map(|row| diff_once(b, row)).collect();
                    let types = rows.iter().map(|v| v.zinnia_type()).collect();
                    return Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() });
                }
            }
            // Scalar-element 1-D: pairwise differences.
            if data.values.len() < 2 {
                let types: Vec<ZinniaType> = vec![];
                return Value::List(CompositeData { elements_type: types, values: vec![], value_id: ValueId::next() });
            }
            let mut out = Vec::with_capacity(data.values.len() - 1);
            for i in 0..data.values.len() - 1 {
                out.push(crate::helpers::value_ops::apply_binary_op(
                    b, "sub", &data.values[i + 1], &data.values[i],
                ));
            }
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: out, value_id: ValueId::next() })
        }
        _ => x.clone(),
    }
}

/// Outer product of two 1-D composites: `result[i][j] = a[i] * b[j]`.
pub fn np_outer(b: &mut IRBuilder, args: &[Value]) -> Value {
    np_outer_op(b, args, "mul")
}

/// Generalized outer-product over a binary op. `np.add.outer`, `np.subtract.outer`,
/// etc. are routed here from the named-attr dispatcher.
pub fn np_outer_op(b: &mut IRBuilder, args: &[Value], op: &str) -> Value {
    let a = match args.first() {
        Some(v) => v,
        None => return Value::None,
    };
    let bv = match args.get(1) {
        Some(v) => v,
        None => return Value::None,
    };
    // Flatten inputs to 1-D (numpy.<op>.outer flattens automatically).
    let a_flat = crate::helpers::composite::flatten_composite(a);
    let b_flat = crate::helpers::composite::flatten_composite(bv);
    let mut rows = Vec::with_capacity(a_flat.len());
    for ai in &a_flat {
        let mut row = Vec::with_capacity(b_flat.len());
        for bj in &b_flat {
            row.push(crate::helpers::value_ops::apply_binary_op(b, op, ai, bj));
        }
        let row_types = row.iter().map(|v| v.zinnia_type()).collect();
        rows.push(Value::List(CompositeData { elements_type: row_types, values: row, value_id: ValueId::next() }));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() })
}

pub fn np_allclose(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    // np.allclose(a, b, rtol=1e-05, atol=1e-08) — check |a - b| <= atol + rtol * |b|
    if args.len() < 2 { return Value::None; }

    // Validate argument types
    fn is_valid_allclose_arg(val: &Value) -> bool {
        matches!(val, Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::List(_) | Value::Tuple(_))
    }
    if !is_valid_allclose_arg(&args[0]) {
        panic!("Unsupported argument type for `lhs`");
    }
    if !is_valid_allclose_arg(&args[1]) {
        panic!("Unsupported argument type for `rhs`");
    }
    if let Some(atol) = kwargs.get("atol").or_else(|| args.get(3)) {
        if !is_valid_allclose_arg(atol) {
            panic!("Unsupported argument type for `atol`");
        }
    }
    if let Some(rtol) = kwargs.get("rtol").or_else(|| args.get(2)) {
        if !is_valid_allclose_arg(rtol) {
            panic!("Unsupported argument type for `rtol`");
        }
    }

    // Extract atol and rtol from kwargs or positional args
    let default_rtol = 1e-5_f64;
    let default_atol = 1e-8_f64;

    fn extract_scalar_float(val: &Value) -> Option<f64> {
        match val {
            Value::Float(s) => s.static_val,
            Value::Integer(s) => s.static_val.map(|i| i as f64),
            Value::List(d) | Value::Tuple(d) if d.values.len() == 1 => {
                extract_scalar_float(&d.values[0])
            }
            _ => None,
        }
    }

    let rtol = kwargs.get("rtol")
        .and_then(|v| extract_scalar_float(v))
        .or_else(|| args.get(2).and_then(|v| extract_scalar_float(v)))
        .unwrap_or(default_rtol);
    let atol = kwargs.get("atol")
        .and_then(|v| extract_scalar_float(v))
        .or_else(|| args.get(3).and_then(|v| extract_scalar_float(v)))
        .unwrap_or(default_atol);

    let a_flat = crate::helpers::composite::flatten_composite(&args[0]);
    let b_flat = crate::helpers::composite::flatten_composite(&args[1]);

    // Handle broadcasting: scalar vs array
    let (a_elems, b_elems) = if a_flat.len() == 1 && b_flat.len() > 1 {
        (vec![a_flat[0].clone(); b_flat.len()], b_flat)
    } else if b_flat.len() == 1 && a_flat.len() > 1 {
        let b_repeated = vec![b_flat[0].clone(); a_flat.len()];
        (a_flat, b_repeated)
    } else {
        (a_flat, b_flat)
    };

    if a_elems.len() != b_elems.len() {
        return b.ir_constant_bool(false);
    }

    // For each element: |a - b| <= atol + rtol * |b|
    // Since we're in ZK, use static evaluation for compile-time known values
    let mut result = b.ir_constant_bool(true);
    for (a_val, b_val) in a_elems.iter().zip(b_elems.iter()) {
        let a_f = a_val.float_val().or_else(|| a_val.int_val().map(|i| i as f64));
        let b_f = b_val.float_val().or_else(|| b_val.int_val().map(|i| i as f64));

        if let (Some(av), Some(bv)) = (a_f, b_f) {
            let diff = (av - bv).abs();
            let threshold = atol + rtol * bv.abs();
            let close = diff <= threshold;
            let close_val = b.ir_constant_bool(close);
            result = b.ir_logical_and(&result, &close_val);
        } else {
            // Dynamic: fall back to exact equality
            let eq = b.ir_equal_i(a_val, b_val);
            result = b.ir_logical_and(&result, &eq);
        }
    }
    result
}

// ── Builtin helpers (single-use) ─────────────────────────────────────

pub fn builtin_range(b: &mut IRBuilder, args: &[Value]) -> Value {
    let (start, stop, step) = match args.len() {
        1 => (0i64, args[0].int_val().unwrap_or(0), 1i64),
        2 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), 1i64),
        3 => (args[0].int_val().unwrap_or(0), args[1].int_val().unwrap_or(0), args[2].int_val().unwrap_or(1)),
        _ => return Value::None,
    };
    if step == 0 { return Value::None; }
    let mut values = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        values.push(b.ir_constant_int(i));
        i += step;
    }
    let types = vec![ZinniaType::Integer; values.len()];
    Value::List(CompositeData { elements_type: types, values, value_id: ValueId::next() })
}

pub fn builtin_len(b: &mut IRBuilder, args: &[Value]) -> Value {
    if let Some(val) = args.first() {
        match val {
            Value::List(data) | Value::Tuple(data) => {
                b.ir_constant_int(data.values.len() as i64)
            }
            _ => b.ir_constant_int(0),
        }
    } else {
        b.ir_constant_int(0)
    }
}

pub fn builtin_enumerate(b: &mut IRBuilder, iter_val: &Value) -> Value {
    match iter_val {
        Value::List(data) | Value::Tuple(data) => {
            let mut result = Vec::new();
            for (i, elem) in data.values.iter().enumerate() {
                let idx = b.ir_constant_int(i as i64);
                result.push(Value::Tuple(CompositeData {
                    elements_type: vec![ZinniaType::Integer, elem.zinnia_type()],
                    values: vec![idx, elem.clone()],

                    value_id: ValueId::next(),
                }));
            }
            let types = result.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: result, value_id: ValueId::next() })
        }
        // P4a follow-up: `enumerate` over a segment-backed StaticArray.
        // We mirror the iteration semantics of `for x in arr` by yielding
        // a `(idx, leaf-or-view)` tuple per outer-axis position.
        Value::StaticArray { shape, .. } => {
            if shape.is_empty() {
                return Value::None;
            }
            let n_iter = shape[0];
            let mut result = Vec::with_capacity(n_iter);
            for i in 0..n_iter {
                let elem = crate::helpers::static_array_read::iter_element(b, iter_val, i);
                let idx = b.ir_constant_int(i as i64);
                result.push(Value::Tuple(CompositeData {
                    elements_type: vec![ZinniaType::Integer, elem.zinnia_type()],
                    values: vec![idx, elem],

                    value_id: ValueId::next(),
                }));
            }
            let types = result.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: result, value_id: ValueId::next() })
        }
        _ => Value::None,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Element-wise math: round / floor / ceil / trunc / reciprocal / where /
// clip. None of these have a dedicated IR primitive yet, so they are
// expressed in terms of existing ops (floor_div, sign, select, etc).
// ────────────────────────────────────────────────────────────────────────

/// Recursively apply `scalar` to every leaf in `val`. Used by all the
/// element-wise wrappers below — keeps the leaf-walking boilerplate in
/// one place. Also reused by `reductions::np_mean` / `np_std` to broadcast
/// a scalar over a composite output.
pub(super) fn vectorize_unary<F: FnMut(&mut IRBuilder, &Value) -> Value>(
    b: &mut IRBuilder,
    val: &Value,
    f: &mut F,
) -> Value {
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let vals: Vec<Value> =
                d.values.iter().map(|v| vectorize_unary(b, v, f)).collect();
            let types = vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: vals,

                value_id: ValueId::next(),
            })
        }
        _ => f(b, val),
    }
}

/// `np.floor(x)` — round towards negative infinity. For floats this is
/// `floor_div(x, 1.0)`; integers and booleans pass through unchanged.
pub fn np_floor(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("floor: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let one = b.ir_constant_float(1.0);
            b.ir_floor_div_f(x, &one)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("floor: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.ceil(x)` — round towards positive infinity. Implemented as
/// `-floor(-x)`.
pub fn np_ceil(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("ceil: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let one = b.ir_constant_float(1.0);
            let neg = b.ir_sub_f(&zero, x);
            let floored = b.ir_floor_div_f(&neg, &one);
            b.ir_sub_f(&zero, &floored)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("ceil: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.trunc(x)` — round towards zero. Implemented as `select(x >= 0,
/// floor(x), ceil(x))`.
pub fn np_trunc(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("trunc: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let one = b.ir_constant_float(1.0);
            // floor branch
            let floor_x = b.ir_floor_div_f(x, &one);
            // ceil branch (= -floor(-x))
            let neg = b.ir_sub_f(&zero, x);
            let neg_floor = b.ir_floor_div_f(&neg, &one);
            let ceil_x = b.ir_sub_f(&zero, &neg_floor);
            let nonneg = b.ir_greater_than_or_equal_f(x, &zero);
            b.ir_select_f(&nonneg, &floor_x, &ceil_x)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("trunc: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.round(x)` — half-away-from-zero (NumPy uses banker's rounding which
/// requires extra primitives we don't have; half-away-from-zero is a
/// reasonable common-case substitute). Implemented as `floor(x + 0.5)` for
/// non-negative x and `-floor(-x + 0.5)` for negative x.
pub fn np_round(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("round: requires an argument");
    vectorize_unary(b, val, &mut |b, x| match x {
        Value::Float(_) => {
            let zero = b.ir_constant_float(0.0);
            let half = b.ir_constant_float(0.5);
            let one = b.ir_constant_float(1.0);
            // pos branch: floor(x + 0.5)
            let pos_in = b.ir_add_f(x, &half);
            let pos_out = b.ir_floor_div_f(&pos_in, &one);
            // neg branch: -floor(-x + 0.5)
            let neg = b.ir_sub_f(&zero, x);
            let neg_in = b.ir_add_f(&neg, &half);
            let neg_floor = b.ir_floor_div_f(&neg_in, &one);
            let neg_out = b.ir_sub_f(&zero, &neg_floor);
            let nonneg = b.ir_greater_than_or_equal_f(x, &zero);
            b.ir_select_f(&nonneg, &pos_out, &neg_out)
        }
        Value::Integer(_) | Value::Boolean(_) => x.clone(),
        _ => panic!("round: unsupported type {:?}", x.zinnia_type()),
    })
}

/// `np.reciprocal(x)` — `1 / x`. Result is always a float.
pub fn np_reciprocal(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("reciprocal: requires an argument");
    vectorize_unary(b, val, &mut |b, x| {
        let xf = match x {
            Value::Float(_) => x.clone(),
            _ => b.ir_float_cast(x),
        };
        let one = b.ir_constant_float(1.0);
        b.ir_div_f(&one, &xf)
    })
}

/// Per-call inputs for the `np.where` strategy set. Carries the already
/// broadcast cond / x / y and the original cond's `value_id` (the anchor
/// the gated preconditions query). `Value` is `Clone`, so this struct can
/// live behind the Phase F framework's
/// `fn(&mut IRBuilder, &Inputs) -> Output` signature.
struct WhereInputs {
    cond_b: Value,
    x_b: Value,
    y_b: Value,
    cond_vid: ValueId,
}

/// Recursive per-element select. Lifted from the original inline `rec`
/// helper inside `np_where` so the generic default lowering can call it
/// behind the strategy framework's fn-pointer signature.
fn where_rec(b: &mut IRBuilder, c: &Value, x: &Value, y: &Value) -> Value {
    match (c, x, y) {
        (
            Value::List(cd) | Value::Tuple(cd),
            Value::List(xd) | Value::Tuple(xd),
            Value::List(yd) | Value::Tuple(yd),
        ) => {
            let vals: Vec<Value> = cd
                .values
                .iter()
                .zip(xd.values.iter())
                .zip(yd.values.iter())
                .map(|((cv, xv), yv)| where_rec(b, cv, xv, yv))
                .collect();
            let types = vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: vals,

                value_id: ValueId::next(),
            })
        }
        _ => crate::helpers::value_ops::select_value(b, c, x, y),
    }
}

/// Strategy A for `where`: `forall_eq_const(cond, 1)` ⇒ every element of
/// `cond` is true ⇒ select picks `x` at every position ⇒ output is `x_b`.
fn lower_where_to_x(_b: &mut IRBuilder, inp: &WhereInputs) -> Value {
    inp.x_b.clone()
}

/// Strategy B for `where`: `forall_eq_const(cond, 0)` ⇒ every element of
/// `cond` is false ⇒ select picks `y` at every position ⇒ output is `y_b`.
fn lower_where_to_y(_b: &mut IRBuilder, inp: &WhereInputs) -> Value {
    inp.y_b.clone()
}

/// Default lowering: existing recursive per-element select. Sound
/// unconditionally.
fn lower_where_generic(b: &mut IRBuilder, inp: &WhereInputs) -> Value {
    where_rec(b, &inp.cond_b, &inp.x_b, &inp.y_b)
}

/// Build the `OpStrategySet` for `np.where` gated on
/// `forall_eq_const(cond, k)` for k ∈ {0, 1}.
fn where_strategy_set(
    cond_vid: ValueId,
) -> crate::optim::OpStrategySet<WhereInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let pred_eq_k = |k: i64| ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(cond_vid)),
            ContractTerm::LitInt(k),
        ],
    };

    OpStrategySet {
        strategies: vec![
            OpStrategy {
                name: "cond_all_true",
                precondition: pred_eq_k(1),
                cost_hint: CostHint::O1,
                lower: lower_where_to_x,
            },
            OpStrategy {
                name: "cond_all_false",
                precondition: pred_eq_k(0),
                cost_hint: CostHint::O1,
                lower: lower_where_to_y,
            },
        ],
        default: lower_where_generic,
    }
}

/// `np.where(cond, x, y)` — element-wise ternary select. All three args
/// are broadcast to a common shape, then a per-element select fires.
///
/// When `cond` carries a `value_id`, the Phase F strategy dispatcher
/// short-circuits to `x` (resp. `y`) if `forall_eq_const(cond, 1)` (resp.
/// `0`) is provable from the visible facts. Otherwise, the default
/// recursive per-element select runs.
pub fn np_where(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() < 3 {
        panic!("where: requires three arguments (cond, x, y)");
    }
    let cond = &args[0];
    let x = &args[1];
    let y = &args[2];
    let cs = crate::helpers::composite::get_composite_shape(cond);
    let xs = crate::helpers::composite::get_composite_shape(x);
    let ys = crate::helpers::composite::get_composite_shape(y);
    // Broadcast cond/x first, then that result with y.
    let cx = crate::helpers::broadcast::broadcast_shapes(&cs, &xs).unwrap_or_else(|| {
        panic!("where: shapes {:?} and {:?} not broadcast compatible", cs, xs)
    });
    let target = crate::helpers::broadcast::broadcast_shapes(&cx, &ys).unwrap_or_else(|| {
        panic!("where: shapes {:?} and {:?} not broadcast compatible", cx, ys)
    });
    let cond_b = crate::helpers::broadcast::materialize_to_shape(cond, &target);
    let x_b = crate::helpers::broadcast::materialize_to_shape(x, &target);
    let y_b = crate::helpers::broadcast::materialize_to_shape(y, &target);
    match cond.value_id() {
        Some(cond_vid) => {
            let inputs = WhereInputs { cond_b, x_b, y_b, cond_vid };
            let set = where_strategy_set(inputs.cond_vid);
            crate::optim::dispatch_strategy(b, "where", &inputs, &set)
        }
        None => where_rec(b, &cond_b, &x_b, &y_b),
    }
}

/// `np.clip(arr, lo, hi)` — element-wise clamp. Implemented as
/// `where(arr < lo, lo, where(arr > hi, hi, arr))`. lo / hi may be scalars
/// or broadcast-compatible arrays.
pub fn np_clip(b: &mut IRBuilder, args: &[Value]) -> Value {
    if args.len() < 3 {
        panic!("clip: requires three arguments (arr, a_min, a_max)");
    }
    let arr = &args[0];
    let lo = &args[1];
    let hi = &args[2];
    // arr.clip(lo, hi) ≡ minimum(maximum(arr, lo), hi)
    let lower = crate::helpers::value_ops::apply_binary_op(b, "lt", arr, lo);
    let after_lower = np_where(b, &[lower, lo.clone(), arr.clone()]);
    let upper = crate::helpers::value_ops::apply_binary_op(b, "gt", &after_lower, hi);
    np_where(b, &[upper, hi.clone(), after_lower])
}
