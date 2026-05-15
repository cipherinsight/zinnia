//! Axis-aware reductions: `reduce_with_axis` (legacy 2-D specialised),
//! `reduce_with_axis_general` (arbitrary-rank), `np_mean` / `np_var` /
//! `np_std`, and the prefix-scan ops `np_cumsum` / `np_cumprod`.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

use super::elementwise::vectorize_unary;

/// Reduce along a specific axis.
/// For a 2D array with axis=0: reduce columns (result is 1D with same ncols)
/// For a 2D array with axis=1: reduce rows (result is 1D with same nrows)
pub fn reduce_with_axis(b: &mut IRBuilder, op: &str, val: &Value, axis: i64) -> Value {
    if let Value::List(outer) | Value::Tuple(outer) = val {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

        if axis == 0 {
            // Reduce along axis 0: for each column position, reduce across rows
            if outer.values.is_empty() { return Value::None; }
            // Get the number of columns from the first row
            if let Value::List(first_row) | Value::Tuple(first_row) = &outer.values[0] {
                let ncols = first_row.values.len();
                let mut results = Vec::new();
                for col in 0..ncols {
                    // Collect all values in this column
                    let mut col_vals = Vec::new();
                    for row in &outer.values {
                        if let Value::List(rd) | Value::Tuple(rd) = row {
                            if col < rd.values.len() {
                                col_vals.push(rd.values[col].clone());
                            }
                        }
                    }
                    let col_list = Value::List(CompositeData {
                        elements_type: col_vals.iter().map(|v| v.zinnia_type()).collect(),
                        values: col_vals,

                        value_id: ValueId::next(),
                    });
                    results.push(crate::helpers::ndarray::builtin_reduce(b, op, &col_list));
                }
                let types = results.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() })
            } else {
                // If first element is scalar, just reduce the whole thing
                crate::helpers::ndarray::builtin_reduce(b, op, val)
            }
        } else if axis == 1 {
            // Reduce along axis 1: for each row, reduce to scalar
            let mut results = Vec::new();
            for row in &outer.values {
                results.push(crate::helpers::ndarray::builtin_reduce(b, op, row));
            }
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() })
        } else {
            crate::helpers::ndarray::builtin_reduce(b, op, val)
        }
    } else {
        crate::helpers::ndarray::builtin_reduce(b, op, val)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Reductions: mean / var / std / cumsum / cumprod (with axis support)
// ────────────────────────────────────────────────────────────────────────

/// Reduce along axis 0 of `items` — i.e. given N input arrays of the same
/// shape, walk them in lockstep and apply `op` element-wise across the N at
/// every leaf position. The result has the inner shape (one rank lower than
/// the outer collection).
fn reduce_along_axis_0(b: &mut IRBuilder, op: &str, items: &[Value]) -> Value {
    if items.is_empty() {
        return Value::None;
    }
    let first = &items[0];
    match first {
        Value::List(d) | Value::Tuple(d) => {
            let inner_len = d.values.len();
            let mut out = Vec::with_capacity(inner_len);
            for i in 0..inner_len {
                let mut inner_items: Vec<Value> = Vec::with_capacity(items.len());
                for it in items {
                    if let Value::List(dd) | Value::Tuple(dd) = it {
                        inner_items.push(dd.values[i].clone());
                    }
                }
                out.push(reduce_along_axis_0(b, op, &inner_items));
            }
            let types = out.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: out,

                value_id: ValueId::next(),
            })
        }
        _ => {
            let lst = Value::List(CompositeData {
                elements_type: items.iter().map(|v| v.zinnia_type()).collect(),
                values: items.to_vec(),

                value_id: ValueId::next(),
            });
            crate::helpers::ndarray::builtin_reduce(b, op, &lst)
        }
    }
}

/// General axis-aware reduction for arbitrary axis. Replaces the old
/// hard-coded axis 0/1 logic.
fn reduce_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: usize) -> Value {
    if axis == 0 {
        if let Value::List(d) | Value::Tuple(d) = val {
            return reduce_along_axis_0(b, op, &d.values);
        }
        return crate::helpers::ndarray::builtin_reduce(b, op, val);
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| reduce_axis_general(b, op, v, axis - 1))
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: new_vals,

                value_id: ValueId::next(),
            })
        }
        _ => crate::helpers::ndarray::builtin_reduce(b, op, val),
    }
}

/// Public entry point preserving the old `reduce_with_axis` name. Resolves
/// negative axes here so callers don't have to.
pub fn reduce_with_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: i64) -> Value {
    let ndim = crate::helpers::composite::get_composite_shape(val).len();
    if ndim == 0 {
        return crate::helpers::ndarray::builtin_reduce(b, op, val);
    }
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "reduce: axis {} is out of bounds for array of rank {}",
            axis, ndim
        );
    }
    reduce_axis_general(b, op, val, resolved as usize)
}

/// Cast a scalar value to float, leaving floats untouched.
fn ensure_scalar_float(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Float(_) => v.clone(),
        _ => b.ir_float_cast(v),
    }
}

/// Per-call inputs for `np_mean`'s whole-array strategy set. Mean's
/// constant lowering is float-typed (numpy semantics: mean returns float),
/// so the lowering doesn't need an `any_float` flag — it always produces
/// `ir_constant_float`.
struct MeanInputs {
    val: Value,
    arr_vid: crate::types::ValueId,
}

fn lower_mean_generic(b: &mut IRBuilder, inputs: &MeanInputs) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(&inputs.val);
    let total: usize = shape.iter().product::<usize>().max(1);
    let total_sum = crate::helpers::ndarray::builtin_reduce(b, "sum", &inputs.val);
    let total_f = ensure_scalar_float(b, &total_sum);
    let n_val = b.ir_constant_float(total as f64);
    b.ir_div_f(&total_f, &n_val)
}

/// Strategy A for `mean`: `forall_eq_const(arr, 0)` ⇒ output is `0.0`.
/// Sound because `(0 + 0 + ... + 0) / N == 0.0` for N >= 1 (guaranteed
/// here by `builtin_reduce`'s empty short-circuit).
fn lower_mean_constant_zero(b: &mut IRBuilder, _inputs: &MeanInputs) -> Value {
    b.ir_constant_float(0.0)
}

/// Strategy B for `mean`: `forall_eq_const(arr, 1)` ⇒ output is `1.0`.
/// Sound because `(1 + 1 + ... + 1) / N == N / N == 1.0` for N >= 1.
fn lower_mean_constant_one(b: &mut IRBuilder, _inputs: &MeanInputs) -> Value {
    b.ir_constant_float(1.0)
}

fn mean_strategy_set(
    arr_vid: crate::types::ValueId,
) -> crate::optim::OpStrategySet<MeanInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let pred_eq_k = |k: i64| ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(arr_vid)),
            ContractTerm::LitInt(k),
        ],
    };

    OpStrategySet {
        strategies: vec![
            OpStrategy {
                name: "forall_eq_const_zero",
                precondition: pred_eq_k(0),
                cost_hint: CostHint::O1,
                lower: lower_mean_constant_zero,
            },
            OpStrategy {
                name: "forall_eq_const_one",
                precondition: pred_eq_k(1),
                cost_hint: CostHint::O1,
                lower: lower_mean_constant_one,
            },
        ],
        default: lower_mean_generic,
    }
}

/// `np.mean(arr, axis=None)` — element-wise mean. With no axis, the result
/// is a scalar; with an axis, the result has rank one less.
pub fn np_mean(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("mean: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let shape = crate::helpers::composite::get_composite_shape(val);
    if let Some(ax) = axis {
        let ndim = shape.len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!("mean: axis {} is out of bounds for array of rank {}", ax, ndim);
        }
        let n = shape[resolved as usize];
        let summed = reduce_axis_general(b, "sum", val, resolved as usize);
        let n_val = b.ir_constant_float(n as f64);
        // Vectorized division
        vectorize_unary(b, &summed, &mut |b, x| {
            let xf = ensure_scalar_float(b, x);
            b.ir_div_f(&xf, &n_val)
        })
    } else {
        let inputs = MeanInputs {
            val: val.clone(),
            arr_vid: val.value_id().unwrap_or_else(ValueId::next),
        };
        match val.value_id() {
            Some(_) => {
                let set = mean_strategy_set(inputs.arr_vid);
                crate::optim::dispatch_strategy(b, "mean", &inputs, &set)
            }
            None => lower_mean_generic(b, &inputs),
        }
    }
}

/// Compute element-wise `(x - m) ** 2` where `m` is broadcast against `x`.
fn squared_deviation(b: &mut IRBuilder, x: &Value, m: &Value) -> Value {
    let diff = crate::helpers::value_ops::apply_binary_op(b, "sub", x, m);
    crate::helpers::value_ops::apply_binary_op(b, "mul", &diff, &diff)
}

/// Materialise the input array's length for the var/std `len_arr` formal:
/// static-array path uses `flatten_composite(arr).len()` as an IR
/// constant, dyn-array path forwards `runtime_length.value_id`.
fn var_std_len_arr_vid(b: &mut IRBuilder, arr: &Value) -> Option<ValueId> {
    match arr {
        Value::DynamicNDArray(d) => Some(d.meta.runtime_length.value_id),
        _ => {
            let n = crate::helpers::composite::flatten_composite(arr).len() as i64;
            b.ir_constant_int(n).value_id()
        }
    }
}

/// `np.var(arr, axis=None)` — population variance (ddof=0). NumPy supports
/// `ddof` but we keep things simple for now and pin ddof=0.
pub fn np_var(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("var: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let mean = np_mean(b, args, kwargs);
    let out = if let Some(ax) = axis {
        // Need to reinsert the reduced axis as length 1 so the broadcast
        // arithmetic works. Easiest: expand_dims at the resolved axis.
        let shape = crate::helpers::composite::get_composite_shape(val);
        let ndim = shape.len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        let axis_const = b.ir_constant_int(resolved);
        let mean_expanded = super::reshaping::np_expand_dims(b, &[mean.clone(), axis_const]);
        let sq = squared_deviation(b, val, &mean_expanded);
        let sq_sum_args = vec![sq.clone()];
        let mut sq_sum_kwargs = HashMap::new();
        sq_sum_kwargs.insert(
            "axis".to_string(),
            b.ir_constant_int(resolved),
        );
        np_mean(b, &sq_sum_args, &sq_sum_kwargs)
    } else {
        let sq = squared_deviation(b, val, &mean);
        np_mean(b, &[sq], &HashMap::new())
    };

    if let (Some(out_vid), Some(len_arr_vid)) = (out.value_id(), var_std_len_arr_vid(b, val)) {
        let mut formals = HashMap::new();
        formals.insert("len_arr".to_string(), len_arr_vid);
        b.fire_contract("var", out_vid, &formals);
    }
    out
}

/// `np.std(arr, axis=None)` — population standard deviation = sqrt(var).
pub fn np_std(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("std: requires an argument");
    let v = np_var(b, args, kwargs);
    let out = vectorize_unary(b, &v, &mut |b, x| {
        let xf = ensure_scalar_float(b, x);
        b.ir_sqrt_f(&xf)
    });

    if let (Some(out_vid), Some(len_arr_vid)) = (out.value_id(), var_std_len_arr_vid(b, val)) {
        let mut formals = HashMap::new();
        formals.insert("len_arr".to_string(), len_arr_vid);
        b.fire_contract("std", out_vid, &formals);
    }
    out
}

/// Inclusive prefix scan along axis 0 of `val`, applying `op` (`add` or
/// `mul`). Used by cumsum/cumprod.
fn cumulative_axis_0(b: &mut IRBuilder, op: &str, val: &Value) -> Value {
    let outer = match val {
        Value::List(d) | Value::Tuple(d) => d,
        _ => return val.clone(),
    };
    if outer.values.is_empty() {
        return val.clone();
    }
    let mut out: Vec<Value> = Vec::with_capacity(outer.values.len());
    out.push(outer.values[0].clone());
    for i in 1..outer.values.len() {
        let prev = out.last().cloned().unwrap();
        let next = crate::helpers::value_ops::apply_binary_op(b, op, &prev, &outer.values[i]);
        out.push(next);
    }
    let types = out.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values: out,

        value_id: ValueId::next(),
    })
}

/// Recursive scan along an arbitrary axis. At axis 0 we run the prefix
/// scan; at axis > 0 we recurse into each outer child.
fn cumulative_axis_general(b: &mut IRBuilder, op: &str, val: &Value, axis: usize) -> Value {
    if axis == 0 {
        return cumulative_axis_0(b, op, val);
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| cumulative_axis_general(b, op, v, axis - 1))
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            Value::List(CompositeData {
                elements_type: types,
                values: new_vals,

                value_id: ValueId::next(),
            })
        }
        _ => val.clone(),
    }
}

/// `np.cumsum(arr, axis=None)` / `np.cumprod(arr, axis=None)`. Without an
/// axis, NumPy flattens first then scans, returning a 1-D result.
pub fn np_cumulative(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    op: &str,
) -> Value {
    let val = args.first().expect("cumulative: requires an argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    if let Some(ax) = axis {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let resolved = if ax < 0 { ndim as i64 + ax } else { ax };
        if resolved < 0 || resolved >= ndim as i64 {
            panic!(
                "{}: axis {} is out of bounds for array of rank {}",
                op, ax, ndim
            );
        }
        cumulative_axis_general(b, op, val, resolved as usize)
    } else {
        // Flatten then scan along the new axis 0.
        let flat = crate::helpers::composite::flatten_composite(val);
        let types: Vec<ZinniaType> = flat.iter().map(|v| v.zinnia_type()).collect();
        let flat_val = Value::List(CompositeData {
            elements_type: types,
            values: flat,

            value_id: ValueId::next(),
        });
        cumulative_axis_0(b, op, &flat_val)
    }
}

pub fn np_cumsum(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    np_cumulative(b, args, kwargs, "add")
}

pub fn np_cumprod(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    np_cumulative(b, args, kwargs, "mul")
}
