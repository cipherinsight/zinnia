//! Array constructors: `np.zeros` / `ones` / `empty` (`np_fill`),
//! `np_fill_like`, `np.identity`, `np.arange`, `np.linspace`. Each
//! constructor has both a fully-static StaticArray path and a
//! bounded-admission DynamicNDArray path that routes through the
//! prove-aware resolver chain.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::optim::resolver::{
    require_provable_static_int, resolve_int_or_bounded, BoundedInt, SiteKind,
};
use crate::types::{Value, ZinniaType};

// ── Numpy-like helpers ────────────────────────────────────────────

/// Fire the `forall_eq_const` content fact for `np.zeros` / `np.ones`
/// (and their `*_like` variants) on the output value. No-op for fill
/// values other than 0 and 1 — `np.full(shape, k)` for arbitrary k is
/// handled by Group 4b's multi-formal contract. No-op when the output
/// Value carries no `value_id` (e.g., the static-array codepath).
fn fire_fill_content_contract(b: &mut IRBuilder, out: &Value, fill_value: i64) {
    let name = match fill_value {
        0 => "zeros_content",
        1 => "ones_content",
        _ => return,
    };
    if let Some(vid) = out.value_id() {
        b.fire_contract(name, vid, &HashMap::new());
    }
}

pub fn np_fill(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>, fill_value: i64) -> Value {
    // np.zeros(shape, dtype=...) / np.ones(shape, dtype=...) / np.empty(shape, dtype=...)
    let arg = match args.first() {
        Some(a) => a,
        None => return Value::None,
    };
    let shape = match arg {
        Value::Integer(_) => {
            // Bounded-aware single-axis dispatch via the prove-aware
            // resolver chain: static_val → resolver range → fact-scan →
            // `IRBuilder::prove` outward-doubling probe. A structurally-
            // or SMT-bounded `k` (e.g., `@requires(lambda x, k: nnz(x) ==
            // k)`, or arithmetic shapes like `k + k <= 20`) routes to a
            // `DynamicNDArray` with `max_length = bound`. See
            // `compiler.consumer-1d-constructor-prove-bounded`.
            use crate::optim::resolver::{resolve_int_or_bounded, BoundedInt};
            match resolve_int_or_bounded(b, arg, SiteKind::ShapeAxis(0), None) {
                BoundedInt::Static(n) => vec![n as usize],
                BoundedInt::Bounded { max, .. } => {
                    // Build a 1-D dyn-ndarray. `arg` is the runtime
                    // active size; `max` is the envelope max from
                    // `resolve_max`. Float dtype falls back to the
                    // existing static path (dyn-ndarray of float is OK).
                    let dtype = if matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Float))) {
                        crate::types::NumberType::Float
                    } else if matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Complex))) {
                        // Complex dyn-ndarray isn't supported; defer to
                        // the static path (which will panic if the
                        // shape isn't statically resolvable).
                        let n: i64 = require_provable_static_int(b, arg, SiteKind::ShapeAxis(0));
                        let fill = b.ir_constant_int(fill_value);
                        let values = vec![fill; n as usize];
                        let out = crate::helpers::static_array::build_static_array_from_flat(
                            b,
                            values,
                            vec![n as usize],
                            crate::types::NumberType::Integer,
                        );
                        fire_fill_content_contract(b, &out, fill_value);
                        return out;
                    } else {
                        crate::types::NumberType::Integer
                    };
                    // dyn_fill_with_active itself fires the content contract
                    // (zeros_content / ones_content) when fill_value is 0 or 1.
                    return crate::ops::dyn_ndarray::constructors::dyn_fill_with_active(
                        b,
                        max as usize,
                        arg.clone(),
                        fill_value,
                        dtype,
                    );
                }
                BoundedInt::Neither => {
                    // Same diagnostic as the prior `require_static_int`
                    // failure mode — keeps the user-facing error message
                    // stable for programs that lack any bound.
                    let _: i64 = require_provable_static_int(b, arg, SiteKind::ShapeAxis(0));
                    unreachable!("require_provable_static_int just panicked above");
                }
            }
        }
        Value::Tuple(data) | Value::List(data) => {
            // Multi-dim bounded path: each axis goes through
            // `resolve_int_or_bounded`. If any axis is bounded (non-static
            // but provably <= some max), promote to a multi-dim dyn-ndarray
            // (uniform fill — buffer is position-independent, so logical-
            // and runtime-strides agree). Complex dtype keeps the static
            // path (dyn-ndarray of Complex is not supported).
            let is_complex = matches!(
                kwargs.get("dtype"),
                Some(Value::Class(ZinniaType::Complex))
            );
            let mut max_shape: Vec<usize> = Vec::with_capacity(data.values.len());
            let mut runtime_shape: Vec<Value> = Vec::with_capacity(data.values.len());
            let mut any_bounded = false;
            for (i, v) in data.values.iter().enumerate() {
                if is_complex {
                    let n: i64 = require_provable_static_int(b, v, SiteKind::ShapeAxis(i));
                    max_shape.push(n.max(0) as usize);
                    runtime_shape.push(v.clone());
                    continue;
                }
                match resolve_int_or_bounded(b, v, SiteKind::ShapeAxis(i), None) {
                    BoundedInt::Static(n) => {
                        max_shape.push(n.max(0) as usize);
                        runtime_shape.push(b.ir_constant_int(n));
                    }
                    BoundedInt::Bounded { max, .. } => {
                        any_bounded = true;
                        max_shape.push(max.max(0) as usize);
                        runtime_shape.push(v.clone());
                    }
                    BoundedInt::Neither => {
                        let _: i64 = require_provable_static_int(b, v, SiteKind::ShapeAxis(i));
                        unreachable!("require_provable_static_int just panicked above");
                    }
                }
            }
            if any_bounded {
                use crate::ops::dyn_ndarray::{
                    constructors::{
                        dyn_from_values_with_active_compact, dyn_from_values_with_active_nd,
                    },
                    value_to_scalar_i64,
                };
                let dtype = if matches!(
                    kwargs.get("dtype"),
                    Some(Value::Class(ZinniaType::Float))
                ) {
                    crate::types::NumberType::Float
                } else {
                    crate::types::NumberType::Integer
                };
                let max_total: usize = max_shape.iter().product();
                let fill_v = match dtype {
                    crate::types::NumberType::Float => b.ir_constant_float(fill_value as f64),
                    crate::types::NumberType::Integer => b.ir_constant_int(fill_value),
                    crate::types::NumberType::Complex => unreachable!(),
                };
                let fill_sv = value_to_scalar_i64(&fill_v);
                let runtime_shape_sv: Vec<crate::types::ScalarValue<i64>> = runtime_shape
                    .iter()
                    .map(value_to_scalar_i64)
                    .collect();
                let mut runtime_length = runtime_shape[0].clone();
                for axis_v in runtime_shape.iter().skip(1) {
                    runtime_length = b.ir_mul_i(&runtime_length, axis_v);
                }
                // Compact-buffer dispatch (multi-dim Case B Tier 1):
                // when the runtime-length product proves to be tighter
                // than `product(max_shape)`, allocate a compact buffer
                // of size `total_bound` instead of `product(max_shape)`.
                // This unlocks programs like `np.zeros((m, n))` with
                // `@requires(m * n <= K)` and `K < m_max * n_max`.
                match resolve_int_or_bounded(
                    b,
                    &runtime_length,
                    SiteKind::ShapeAxis(0),
                    None,
                ) {
                    BoundedInt::Static(n) => {
                        let n = n.max(0) as usize;
                        if n < max_total {
                            let out = dyn_from_values_with_active_compact(
                                b,
                                fill_sv,
                                max_shape,
                                runtime_shape_sv,
                                n,
                                dtype,
                            );
                            fire_fill_content_contract(b, &out, fill_value);
                            return out;
                        }
                    }
                    BoundedInt::Bounded { max, .. } => {
                        let total_bound = (max.max(0) as usize).min(max_total);
                        if total_bound < max_total {
                            let out = dyn_from_values_with_active_compact(
                                b,
                                fill_sv,
                                max_shape,
                                runtime_shape_sv,
                                total_bound,
                                dtype,
                            );
                            fire_fill_content_contract(b, &out, fill_value);
                            return out;
                        }
                    }
                    BoundedInt::Neither => {}
                }
                let values = vec![fill_sv; max_total];
                let out = dyn_from_values_with_active_nd(
                    b,
                    values,
                    max_shape,
                    runtime_shape_sv,
                    runtime_length,
                    dtype,
                );
                fire_fill_content_contract(b, &out, fill_value);
                return out;
            }
            max_shape
        }
        _ => panic!("np.zeros/ones/empty: shape must be int, tuple, or list (got {:?})", arg.zinnia_type()),
    };
    let total: usize = shape.iter().product();

    // Complex dtype: produce a Complex StaticArray (dual-segment).
    if matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Complex))) {
        let real_fill = b.ir_constant_float(fill_value as f64);
        let imag_fill = b.ir_constant_float(0.0);
        let reals = vec![real_fill; total];
        let imags = vec![imag_fill; total];
        let out = crate::helpers::static_array::build_static_array_from_flat_complex(
            b, reals, imags, shape,
        );
        fire_fill_content_contract(b, &out, fill_value);
        return out;
    }

    let use_float = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Float)));
    let (fill, dtype) = if use_float {
        (b.ir_constant_float(fill_value as f64), crate::types::NumberType::Float)
    } else {
        (b.ir_constant_int(fill_value), crate::types::NumberType::Integer)
    };
    let values = vec![fill; total];
    // P1 segarr-foundation: numeric constructors emit Value::StaticArray.
    let out = crate::helpers::static_array::build_static_array_from_flat(b, values, shape, dtype);
    fire_fill_content_contract(b, &out, fill_value);
    out
}

pub fn np_fill_like(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>, fill_value: i64) -> Value {
    // np.empty_like(x) / np.zeros_like(x) / np.ones_like(x)
    // Shape is taken from x; dtype defaults to x's dtype, overridable via dtype= kwarg.
    let x = match args.first() {
        Some(v) => v,
        None => return Value::None,
    };
    let shape = if let Value::StaticArray { shape, .. } = x {
        shape.clone()
    } else {
        crate::helpers::composite::get_composite_shape(x)
    };
    // Detect Complex dtype: explicit dtype=complex or x is a Complex array.
    let is_complex = if let Some(Value::Class(ZinniaType::Complex)) = kwargs.get("dtype") {
        true
    } else if matches!(kwargs.get("dtype"), Some(Value::Class(_))) {
        false
    } else {
        match x {
            Value::StaticArray { dtype: crate::types::NumberType::Complex, .. } => true,
            _ => crate::helpers::composite::flatten_composite(x)
                .iter()
                .any(|v| matches!(v.zinnia_type(), ZinniaType::Complex)),
        }
    };
    let total: usize = shape.iter().product();
    if is_complex {
        let real_fill = b.ir_constant_float(fill_value as f64);
        let imag_fill = b.ir_constant_float(0.0);
        let reals = vec![real_fill; total];
        let imags = vec![imag_fill; total];
        let out = crate::helpers::static_array::build_static_array_from_flat_complex(
            b, reals, imags, shape,
        );
        fire_fill_content_contract(b, &out, fill_value);
        return out;
    }
    let use_float = if let Some(Value::Class(ZinniaType::Float)) = kwargs.get("dtype") {
        true
    } else if matches!(kwargs.get("dtype"), Some(Value::Class(_))) {
        false
    } else {
        match x {
            Value::StaticArray { dtype: crate::types::NumberType::Float, .. } => true,
            _ => crate::helpers::composite::flatten_composite(x)
                .iter()
                .any(|v| matches!(v.zinnia_type(), ZinniaType::Float)),
        }
    };
    let (fill, dtype) = if use_float {
        (b.ir_constant_float(fill_value as f64), crate::types::NumberType::Float)
    } else {
        (b.ir_constant_int(fill_value), crate::types::NumberType::Integer)
    };
    let values = vec![fill; total];
    // P1 segarr-foundation: numeric constructors emit Value::StaticArray.
    let out = crate::helpers::static_array::build_static_array_from_flat(b, values, shape, dtype);
    fire_fill_content_contract(b, &out, fill_value);
    out
}

pub fn np_identity(b: &mut IRBuilder, args: &[Value]) -> Value {
    // Behaviour change vs. earlier revisions: previously the code did
    // `args.first().and_then(|a| a.int_val()).unwrap_or(0)`, silently
    // producing a length-0 array for any non-literal argument. That was a
    // pre-existing bug (compiler.consumer-deferred-bounded-sweep). The
    // bounded path promotes those programs to a 2-D `DynamicNDArray`;
    // programs whose `N` lacks any provable static / bounded interpretation
    // now panic loudly via `require_provable_static_int` instead of
    // compiling wrong.
    use crate::ops::dyn_ndarray::{
        constructors::dyn_from_values_with_active_nd, value_to_scalar_i64,
    };

    let n_arg = args.first().expect("identity: N argument required");
    match resolve_int_or_bounded(b, n_arg, SiteKind::ShapeAxis(0), None) {
        BoundedInt::Static(n) => {
            let n = n.max(0) as usize;
            let zero = b.ir_constant_int(0);
            let one = b.ir_constant_int(1);
            let mut flat = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    flat.push(if i == j { one.clone() } else { zero.clone() });
                }
            }
            let out = crate::helpers::static_array::build_static_array_from_flat(
                b,
                flat,
                vec![n, n],
                crate::types::NumberType::Integer,
            );
            if let Some(vid) = out.value_id() {
                b.fire_contract("identity_content", vid, &HashMap::new());
            }
            out
        }
        BoundedInt::Bounded { max, .. } => {
            // Natural-padding works because slot `i * N_max + j` in the
            // buffer is `1` iff `i == j`, which is the right value for any
            // valid index `(i, j)` with `i, j < N` — independent of N.
            let n_max = max.max(0) as usize;
            let zero = b.ir_constant_int(0);
            let one = b.ir_constant_int(1);
            let mut values: Vec<crate::types::ScalarValue<i64>> =
                Vec::with_capacity(n_max * n_max);
            for i in 0..n_max {
                for j in 0..n_max {
                    let v = if i == j { one.clone() } else { zero.clone() };
                    values.push(value_to_scalar_i64(&v));
                }
            }
            let n_sv = value_to_scalar_i64(n_arg);
            let runtime_length = b.ir_mul_i(n_arg, n_arg);
            let result = dyn_from_values_with_active_nd(
                b,
                values,
                vec![n_max, n_max],
                vec![n_sv.clone(), n_sv],
                runtime_length,
                crate::types::NumberType::Integer,
            );
            // Fact: runtime_length == N * N.
            let runtime_length_vid = match &result {
                Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
                _ => unreachable!(),
            };
            let n_vid = n_arg.value_id().expect("identity: bounded N must be an SSA scalar");
            let mut formals = std::collections::HashMap::new();
            formals.insert("N".to_string(), n_vid);
            b.fire_contract("dyn_identity", runtime_length_vid, &formals);
            if let Some(vid) = result.value_id() {
                b.fire_contract("identity_content", vid, &HashMap::new());
            }
            result
        }
        BoundedInt::Neither => {
            let _: i64 = require_provable_static_int(b, n_arg, SiteKind::ShapeAxis(0));
            unreachable!("require_provable_static_int just panicked above");
        }
    }
}

pub fn np_arange(b: &mut IRBuilder, args: &[Value]) -> Value {
    // np.arange always returns a numeric ndarray — emit as StaticArray
    // when fully static, or DynamicNDArray when the stop is symbolic
    // but bounded via the prove-aware resolver chain.
    //
    // Behaviour change vs. earlier revisions: previously the code did
    // `args[i].int_val().unwrap_or(0)`, silently producing a length-0
    // array for any non-literal argument. That was a pre-existing bug
    // (compiler.consumer-arange-tile-prove-bounded). The bounded path
    // promotes those programs to a `DynamicNDArray`; programs whose
    // arguments lack any provable static / bounded interpretation now
    // panic loudly via `require_provable_static_int` instead of compiling
    // wrong.
    use crate::ops::dyn_ndarray::{constructors::dyn_from_values_with_active, value_to_scalar_i64};

    match args.len() {
        1 => {
            let stop_val = &args[0];
            match resolve_int_or_bounded(b, stop_val, SiteKind::RangeStop, None) {
                BoundedInt::Static(stop) => arange_static(b, 0, stop, 1),
                BoundedInt::Bounded { max, .. } => {
                    let n_max = max.max(0) as usize;
                    let values: Vec<crate::types::ScalarValue<i64>> = (0..n_max)
                        .map(|i| {
                            let v = b.ir_constant_int(i as i64);
                            value_to_scalar_i64(&v)
                        })
                        .collect();
                    let result = dyn_from_values_with_active(
                        b,
                        values,
                        stop_val.clone(),
                        crate::types::NumberType::Integer,
                    );
                    // Fact: runtime_length == stop - 0 (instantiates to
                    // `runtime_length == stop` after Z3 simplification).
                    let runtime_length_vid = match &result {
                        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
                        _ => unreachable!(),
                    };
                    let start_vid = b.ir_constant_int(0).value_id().unwrap();
                    let stop_vid = stop_val.value_id().unwrap();
                    let mut formals = std::collections::HashMap::new();
                    formals.insert("start".to_string(), start_vid);
                    formals.insert("stop".to_string(), stop_vid);
                    b.fire_contract("dyn_arange", runtime_length_vid, &formals);
                    // 1-arg form: step is implicit 1 ⇒ always ascending ⇒
                    // is_sorted(out) holds. Fire on the array's value_id
                    // (not the length-bearing scalar).
                    if let Some(vid) = result.value_id() {
                        b.fire_contract(
                            "arange_is_sorted",
                            vid,
                            &std::collections::HashMap::new(),
                        );
                    }
                    result
                }
                BoundedInt::Neither => {
                    let _: i64 = require_provable_static_int(b, stop_val, SiteKind::RangeStop);
                    unreachable!("require_provable_static_int just panicked above");
                }
            }
        }
        2 => {
            // start must be a static int; stop may be bounded.
            let start_val = &args[0];
            let stop_val = &args[1];
            let start: i64 = require_provable_static_int(b, start_val, SiteKind::RangeStart);
            match resolve_int_or_bounded(b, stop_val, SiteKind::RangeStop, None) {
                BoundedInt::Static(stop) => arange_static(b, start, stop, 1),
                BoundedInt::Bounded { max, .. } => {
                    let n_max = (max - start).max(0) as usize;
                    let values: Vec<crate::types::ScalarValue<i64>> = (0..n_max)
                        .map(|i| {
                            let v = b.ir_constant_int(start + i as i64);
                            value_to_scalar_i64(&v)
                        })
                        .collect();
                    let start_constant = b.ir_constant_int(start);
                    let runtime_length = b.ir_sub_i(stop_val, &start_constant);
                    let start_vid = start_constant.value_id().unwrap();
                    let stop_vid = stop_val.value_id().unwrap();
                    let result = dyn_from_values_with_active(
                        b,
                        values,
                        runtime_length,
                        crate::types::NumberType::Integer,
                    );
                    // Fact: runtime_length == stop - start.
                    let runtime_length_vid = match &result {
                        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
                        _ => unreachable!(),
                    };
                    let mut formals = std::collections::HashMap::new();
                    formals.insert("start".to_string(), start_vid);
                    formals.insert("stop".to_string(), stop_vid);
                    b.fire_contract("dyn_arange", runtime_length_vid, &formals);
                    // 2-arg form: step is implicit 1 ⇒ always ascending.
                    if let Some(vid) = result.value_id() {
                        b.fire_contract(
                            "arange_is_sorted",
                            vid,
                            &std::collections::HashMap::new(),
                        );
                    }
                    result
                }
                BoundedInt::Neither => {
                    let _: i64 = require_provable_static_int(b, stop_val, SiteKind::RangeStop);
                    unreachable!("require_provable_static_int just panicked above");
                }
            }
        }
        3 => {
            // Symbolic-step support stays out of scope: `start` and `step`
            // must be literal. `stop` may be bounded (the bounded path
            // computes runtime_length = (stop - start) / step for positive
            // step, mirroring numpy's `(stop - start + step - 1) // step`
            // truncation-towards-zero for non-aligned `stop`).
            let start: i64 = require_provable_static_int(b, &args[0], SiteKind::RangeStart);
            let step: i64 = require_provable_static_int(b, &args[2], SiteKind::RangeStep);
            let stop_val = &args[1];
            match resolve_int_or_bounded(b, stop_val, SiteKind::RangeStop, None) {
                BoundedInt::Static(stop) => arange_static(b, start, stop, step),
                BoundedInt::Bounded { max, .. } => {
                    if step == 0 {
                        return Value::None;
                    }
                    if step < 0 {
                        // Negative-step bounded form is out of scope.
                        let _: i64 = require_provable_static_int(
                            b,
                            stop_val,
                            SiteKind::RangeStop,
                        );
                        unreachable!("require_provable_static_int just panicked above");
                    }
                    // Positive step: len_max = ceildiv(max - start, step)
                    let span_max = (max - start).max(0);
                    let n_max = ((span_max + step - 1) / step).max(0) as usize;
                    let values: Vec<crate::types::ScalarValue<i64>> = (0..n_max)
                        .map(|i| {
                            let v = b.ir_constant_int(start + (i as i64) * step);
                            value_to_scalar_i64(&v)
                        })
                        .collect();
                    let start_const = b.ir_constant_int(start);
                    let step_const = b.ir_constant_int(step);
                    let span = b.ir_sub_i(stop_val, &start_const);
                    let one = b.ir_constant_int(1);
                    let step_minus_one = b.ir_sub_i(&step_const, &one);
                    let span_plus = b.ir_add_i(&span, &step_minus_one);
                    let runtime_length = b.ir_div_i(&span_plus, &step_const);
                    let result = dyn_from_values_with_active(
                        b,
                        values,
                        runtime_length,
                        crate::types::NumberType::Integer,
                    );
                    // Bounded 3-arg path reaches here only when `step > 0`
                    // (the `step < 0` branch above already panicked), so the
                    // direction check has effectively been made — fire
                    // unconditionally here.
                    if let Some(vid) = result.value_id() {
                        b.fire_contract(
                            "arange_is_sorted",
                            vid,
                            &std::collections::HashMap::new(),
                        );
                    }
                    result
                }
                BoundedInt::Neither => {
                    let _: i64 = require_provable_static_int(b, stop_val, SiteKind::RangeStop);
                    unreachable!("require_provable_static_int just panicked above");
                }
            }
        }
        _ => Value::None,
    }
}

fn arange_static(b: &mut IRBuilder, start: i64, stop: i64, step: i64) -> Value {
    if step == 0 {
        return Value::None;
    }
    let mut values = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        values.push(b.ir_constant_int(i));
        i += step;
    }
    let len = values.len();
    let result = crate::helpers::static_array::build_static_array_from_flat(
        b,
        values,
        vec![len],
        crate::types::NumberType::Integer,
    );
    // Fire `is_sorted(out)` when the step is positive (ascending). The
    // 3-arg call site above fires its own conditional and bypasses this
    // helper for the descending case via panic, but `arange_static` is
    // also reached by the 1-arg / 2-arg static branches (step is
    // implicitly 1) and by the 3-arg static-stop branch (step is the
    // user's literal). Gating here keeps the soundness check local. The
    // `Value::StaticArray` returned by `build_static_array_from_flat`
    // does not carry a value_id today, so the fire is a no-op for the
    // static composite — but if a future revision attaches an identity
    // to static arrays, the fact will start landing automatically.
    if step > 0 {
        if let Some(vid) = result.value_id() {
            b.fire_contract(
                "arange_is_sorted",
                vid,
                &std::collections::HashMap::new(),
            );
        }
    }
    result
}

pub fn np_linspace(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    if args.len() < 2 { return Value::None; }
    let start = args[0].float_val().or_else(|| args[0].int_val().map(|v| v as f64)).unwrap_or(0.0);
    let stop = args[1].float_val().or_else(|| args[1].int_val().map(|v| v as f64)).unwrap_or(0.0);
    let endpoint = kwargs.get("endpoint").and_then(|v| v.bool_val()).unwrap_or(true);
    let use_int = matches!(kwargs.get("dtype"), Some(Value::Class(ZinniaType::Integer)));
    let dtype = if use_int { crate::types::NumberType::Integer } else { crate::types::NumberType::Float };

    // `num` may be a positional arg, a kwarg, or absent (default 50). Try
    // the bounded-aware dispatch on whichever Value source is provided;
    // fall back to literal 50 when neither path supplies one.
    let num_val: Option<&Value> = args.get(2).or_else(|| kwargs.get("num"));

    if let Some(num_arg) = num_val {
        match resolve_int_or_bounded(b, num_arg, SiteKind::LinspaceNum, None) {
            BoundedInt::Static(n) => {
                return np_linspace_static(b, start, stop, n.max(0) as usize, endpoint, use_int, dtype);
            }
            BoundedInt::Bounded { max, .. } => {
                // Soundness guard: `denom = num - 1` (endpoint=true) or
                // `denom = num` (endpoint=false) feeds `ir_div_f`. To avoid
                // division by zero, require `num >= 2` for endpoint and
                // `num >= 1` for !endpoint. If not provable, refuse rather
                // than silently producing NaN.
                use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
                use crate::optim::prove::ProveOutcome;
                let needed_min: i64 = if endpoint { 2 } else { 1 };
                let num_vid = num_arg
                    .value_id()
                    .expect("linspace: bounded num must be an SSA scalar");
                let ge_term = ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
                    rhs: Box::new(ContractTerm::LitInt(needed_min)),
                };
                if !matches!(b.prove(&ge_term), ProveOutcome::Proved) {
                    panic!(
                        "linspace: bounded `num` is not provably >= {} (required to avoid division by zero with endpoint={}); supply tighter @requires facts or use a literal num",
                        needed_min, endpoint,
                    );
                }
                let num_max = max.max(0) as usize;
                return np_linspace_bounded(
                    b, start, stop, num_arg, num_max, endpoint, use_int,
                );
            }
            BoundedInt::Neither => {
                // Fall through to the legacy default-50 path, preserving
                // backward compatibility: if `num` is unresolvable we
                // can't admit it as bounded either, so try `int_val()`
                // one last time (handles the constant-folded fallback).
                if let Some(n) = num_arg.int_val() {
                    return np_linspace_static(b, start, stop, n.max(0) as usize, endpoint, use_int, dtype);
                }
                let _: i64 = require_provable_static_int(b, num_arg, SiteKind::LinspaceNum);
                unreachable!("require_provable_static_int just panicked above");
            }
        }
    }

    np_linspace_static(b, start, stop, 50, endpoint, use_int, dtype)
}

fn np_linspace_static(
    b: &mut IRBuilder,
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    use_int: bool,
    dtype: crate::types::NumberType,
) -> Value {
    if num == 0 {
        return crate::helpers::static_array::build_static_array_from_flat(b, vec![], vec![0], dtype);
    }
    if num == 1 {
        let v = if use_int { b.ir_constant_int(start as i64) } else { b.ir_constant_float(start) };
        return crate::helpers::static_array::build_static_array_from_flat(b, vec![v], vec![1], dtype);
    }

    let divisor = if endpoint { (num - 1) as f64 } else { num as f64 };
    let step = (stop - start) / divisor;
    let mut values = Vec::new();
    for i in 0..num {
        let fval = start + step * i as f64;
        if use_int {
            values.push(b.ir_constant_int(fval as i64));
        } else {
            values.push(b.ir_constant_float(fval));
        }
    }
    let len = values.len();
    let result =
        crate::helpers::static_array::build_static_array_from_flat(b, values, vec![len], dtype);
    // Fire `is_sorted(out)` only when start <= stop (ascending or equal).
    // The fully-static StaticArray path does not carry a value_id today,
    // so the fire is a no-op for the static composite — kept for
    // symmetry and forward-compatibility with consumers that one day
    // attach identity to static arrays.
    if start <= stop {
        if let Some(vid) = result.value_id() {
            b.fire_contract(
                "linspace_is_sorted",
                vid,
                &std::collections::HashMap::new(),
            );
        }
    }
    result
}

fn np_linspace_bounded(
    b: &mut IRBuilder,
    start: f64,
    stop: f64,
    num_arg: &Value,
    num_max: usize,
    endpoint: bool,
    use_int: bool,
) -> Value {
    use crate::ops::dyn_ndarray::{
        constructors::dyn_from_values_with_active, value_to_scalar_i64,
    };

    // Float dtype for the output buffer; integer dtype casts on write.
    let out_dtype = if use_int {
        crate::types::NumberType::Integer
    } else {
        crate::types::NumberType::Float
    };

    if num_max == 0 {
        let runtime_length = num_arg.clone();
        return dyn_from_values_with_active(b, Vec::new(), runtime_length, out_dtype);
    }

    // Compute symbolic step = (stop - start) / denom_f.
    // Soundness (guarded above): we have `num >= needed_min` (2 if
    // endpoint, else 1), so denom > 0 — no division by zero.
    let one_int = b.ir_constant_int(1);
    let denom_i = if endpoint {
        b.ir_sub_i(num_arg, &one_int)
    } else {
        num_arg.clone()
    };
    let denom_f = b.ir_float_cast(&denom_i);
    let start_const = b.ir_constant_float(start);
    let stop_const = b.ir_constant_float(stop);
    let span = b.ir_sub_f(&stop_const, &start_const);
    let step = b.ir_div_f(&span, &denom_f);

    // Allocate output segment of `num_max` slots and overwrite the active
    // region via per-cell symbolic writes. Slots beyond `runtime_length`
    // are never read by the subscript machinery.
    let default_v = match out_dtype {
        crate::types::NumberType::Float => b.ir_constant_float(0.0),
        crate::types::NumberType::Integer => b.ir_constant_int(0),
        crate::types::NumberType::Complex => unreachable!(),
    };
    let default_sv = value_to_scalar_i64(&default_v);
    let init = vec![default_sv; num_max];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &init, out_dtype);

    for i in 0..num_max {
        let i_const_f = b.ir_constant_float(i as f64);
        let offset = b.ir_mul_f(&i_const_f, &step);
        let val_f = b.ir_add_f(&start_const, &offset);
        let val = if use_int { b.ir_int_cast(&val_f) } else { val_f };
        let i_const_i = b.ir_constant_int(i as i64);
        b.ir_write_memory(segment_id, &i_const_i, &val);
    }

    // 1-D dyn-ndarray with runtime_length = num.
    let runtime_length_sv = value_to_scalar_i64(num_arg);
    let runtime_length_vid = runtime_length_sv.value_id;
    let logical_shape = vec![num_max];
    let envelope =
        crate::types::Envelope::from_static_shape(&mut b.dim_table, &logical_shape);

    // Fact: runtime_length == num.
    let num_vid = num_arg.value_id().expect("linspace: bounded num must be an SSA scalar");
    let mut formals = std::collections::HashMap::new();
    formals.insert("num".to_string(), num_vid);
    b.fire_contract("dyn_linspace", runtime_length_vid, &formals);

    let result = Value::DynamicNDArray(crate::types::DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: crate::types::DynArrayMeta {
            logical_shape,
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: runtime_length_sv.clone(),
            runtime_rank: crate::types::ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_length_sv],
            runtime_strides: vec![crate::types::ScalarValue::new(Some(1), None)],
            runtime_offset: crate::types::ScalarValue::new(Some(0), None),
        },
        value_id: crate::types::ValueId::next(),
    });

    // Fire `is_sorted(out)` on the dyn-ndarray's value_id when start <= stop
    // (ascending or all-equal). Anchored on the array's value_id, not the
    // length-bearing scalar — these are two distinct facts on two distinct
    // SSA identities. Soundness: `start > stop` would produce a descending
    // sequence; we simply skip the fire in that case (no false claim).
    if start <= stop {
        if let Some(vid) = result.value_id() {
            b.fire_contract(
                "linspace_is_sorted",
                vid,
                &std::collections::HashMap::new(),
            );
        }
    }

    result
}
