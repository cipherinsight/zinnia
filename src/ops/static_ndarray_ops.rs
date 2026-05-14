use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::optim::resolver::{
    require_provable_static_int, resolve_int_or_bounded, BoundedInt, SiteKind,
};
use crate::types::{CompositeData, Value, ValueId, ZinniaType};

pub fn matmul(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs_shape = crate::helpers::composite::get_composite_shape(lhs);
    let rhs_shape = crate::helpers::composite::get_composite_shape(rhs);

    // Scalar case
    if lhs_shape.is_empty() || rhs_shape.is_empty() {
        return crate::helpers::value_ops::apply_binary_op(b, "mul", lhs, rhs);
    }

    let lhs_cols = *lhs_shape.last().unwrap();
    let rhs_rows = if rhs_shape.len() >= 1 { rhs_shape[0] } else { 1 };

    if lhs_cols != rhs_rows {
        panic!("their shapes are not multiply compatible: ({}) and ({})",
            lhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
            rhs_shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "),
        );
    }

    // Determine if we should use float ops (if either operand has float elements)
    let lhs_flat = crate::helpers::composite::flatten_composite(lhs);
    let rhs_flat = crate::helpers::composite::flatten_composite(rhs);
    let use_float = lhs_flat.iter().any(|v| matches!(v, Value::Float(_)))
        || rhs_flat.iter().any(|v| matches!(v, Value::Float(_)));
    let use_complex = lhs_flat.iter().any(|v| matches!(v, Value::Complex { .. }))
        || rhs_flat.iter().any(|v| matches!(v, Value::Complex { .. }));

    // Phase F strategy dispatch: if either operand is provably all-zeros
    // (forall_eq_const(_, 0)), the matmul output is the zero matrix of the
    // broadcast shape. Falls through to the generic body when no fact is
    // visible or when either operand lacks a `value_id`.
    if let (Some(lhs_vid), Some(rhs_vid)) = (lhs.value_id(), rhs.value_id()) {
        let out_shape = compute_matmul_output_shape(&lhs_shape, &rhs_shape);
        let inputs = MatmulInputs {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            out_shape,
            use_float,
            use_complex,
        };
        let set = matmul_strategy_set(lhs_vid, rhs_vid);
        return crate::optim::dispatch_strategy(b, "matmul", &inputs, &set);
    }

    matmul_generic(b, lhs, rhs)
}

/// Compute the matmul broadcast output shape from `lhs_shape` and
/// `rhs_shape`. Assumes the shape-compatibility check already passed:
/// 1D@1D → `[]`, 2D@1D → `[lhs_shape[0]]`, 1D@2D → `[rhs_shape[1]]`,
/// 2D@2D → `[lhs_shape[0], rhs_shape[1]]`.
fn compute_matmul_output_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> Vec<usize> {
    match (lhs_shape.len(), rhs_shape.len()) {
        (1, 1) => vec![],
        (2, 1) => vec![lhs_shape[0]],
        (1, 2) => vec![rhs_shape[1]],
        (2, 2) => vec![lhs_shape[0], rhs_shape[1]],
        _ => panic!(
            "matmul: unsupported shape combination {:?} @ {:?}",
            lhs_shape, rhs_shape
        ),
    }
}

/// Per-call inputs for the `matmul` strategy set. Carries the original
/// lhs/rhs Values (used by the generic default), the precomputed output
/// shape, and dtype flags consumed by the zero-output lowering.
struct MatmulInputs {
    lhs: Value,
    rhs: Value,
    out_shape: Vec<usize>,
    use_float: bool,
    use_complex: bool,
}

/// Build a nested `Value::List` of zeros with the requested shape. Scalar
/// shape `[]` returns a plain int/float/complex zero. For non-empty
/// shapes, the outermost composite's `value_id` is published via the
/// `zeros_content` ensure so downstream consumers see
/// `forall_eq_const(out, 0)` planted on the output.
fn build_zeros_of_shape(
    b: &mut IRBuilder,
    shape: &[usize],
    use_float: bool,
    use_complex: bool,
) -> Value {
    let scalar_zero = |b: &mut IRBuilder| -> Value {
        if use_complex {
            let zero_re = b.ir_constant_float(0.0);
            let zero_im = b.ir_constant_float(0.0);
            let r = match zero_re {
                Value::Float(s) => s,
                _ => unreachable!(),
            };
            let i = match zero_im {
                Value::Float(s) => s,
                _ => unreachable!(),
            };
            Value::Complex { real: r, imag: i }
        } else if use_float {
            b.ir_constant_float(0.0)
        } else {
            b.ir_constant_int(0)
        }
    };
    if shape.is_empty() {
        return scalar_zero(b);
    }
    fn build_level(
        b: &mut IRBuilder,
        shape: &[usize],
        leaf: &mut dyn FnMut(&mut IRBuilder) -> Value,
    ) -> Value {
        if shape.len() == 1 {
            let vals: Vec<Value> = (0..shape[0]).map(|_| leaf(b)).collect();
            let types = vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: vals,
                value_id: ValueId::next(),
            });
        }
        let inner: Vec<Value> = (0..shape[0])
            .map(|_| build_level(b, &shape[1..], leaf))
            .collect();
        let types = inner.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values: inner,
            value_id: ValueId::next(),
        })
    }
    let mut leaf = scalar_zero;
    let out = build_level(b, shape, &mut leaf);
    if let Some(vid) = out.value_id() {
        b.fire_contract("zeros_content", vid, &HashMap::new());
    }
    out
}

/// Strategy lowering shared by `lhs_all_zero` and `rhs_all_zero`: emit a
/// zeros composite of the precomputed output shape. Sound because every
/// product in a matmul sum is zero when any factor of every pair is zero.
fn lower_matmul_zero(b: &mut IRBuilder, inp: &MatmulInputs) -> Value {
    build_zeros_of_shape(b, &inp.out_shape, inp.use_float, inp.use_complex)
}

/// Default lowering: existing generic matmul body. Sound unconditionally.
fn lower_matmul_generic(b: &mut IRBuilder, inp: &MatmulInputs) -> Value {
    matmul_generic(b, &inp.lhs, &inp.rhs)
}

/// Strategy lowering for `is_identity(lhs)`: when the lhs is provably the
/// identity matrix, `I @ B = B`. Matmul's shape check has already verified
/// `lhs.cols == rhs.rows`, and `is_identity` constrains lhs to be N×N, so
/// rhs's shape matches the expected output shape.
fn lower_matmul_lhs_identity_returns_rhs(_b: &mut IRBuilder, inp: &MatmulInputs) -> Value {
    inp.rhs.clone()
}

/// Strategy lowering for `is_identity(rhs)`: when the rhs is provably the
/// identity matrix, `A @ I = A`. Symmetric to the lhs case.
fn lower_matmul_rhs_identity_returns_lhs(_b: &mut IRBuilder, inp: &MatmulInputs) -> Value {
    inp.lhs.clone()
}

/// Build the `OpStrategySet` for `matmul` gated on
/// `forall_eq_const(lhs, 0)` and `forall_eq_const(rhs, 0)`. Either match
/// short-circuits to a zeros composite.
fn matmul_strategy_set(
    lhs_vid: ValueId,
    rhs_vid: ValueId,
) -> crate::optim::OpStrategySet<MatmulInputs, Value> {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::{CostHint, OpStrategy, OpStrategySet};

    let pred_eq_zero = |vid: ValueId| ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(vid)),
            ContractTerm::LitInt(0),
        ],
    };

    let pred_is_identity = |vid: ValueId| ContractTerm::PredicateApp {
        kind: "is_identity".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(vid))],
    };

    OpStrategySet {
        strategies: vec![
            OpStrategy {
                name: "lhs_all_zero",
                precondition: pred_eq_zero(lhs_vid),
                cost_hint: CostHint::O1,
                lower: lower_matmul_zero,
            },
            OpStrategy {
                name: "rhs_all_zero",
                precondition: pred_eq_zero(rhs_vid),
                cost_hint: CostHint::O1,
                lower: lower_matmul_zero,
            },
            OpStrategy {
                name: "lhs_is_identity",
                precondition: pred_is_identity(lhs_vid),
                cost_hint: CostHint::O1,
                lower: lower_matmul_lhs_identity_returns_rhs,
            },
            OpStrategy {
                name: "rhs_is_identity",
                precondition: pred_is_identity(rhs_vid),
                cost_hint: CostHint::O1,
                lower: lower_matmul_rhs_identity_returns_lhs,
            },
        ],
        default: lower_matmul_generic,
    }
}

/// Generic matmul body, lifted from the original `matmul` so it can run
/// as both the strategy framework's `default` lowering and the
/// no-value_id fall-through inside `matmul`.
fn matmul_generic(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> Value {
    let lhs_shape = crate::helpers::composite::get_composite_shape(lhs);
    let rhs_shape = crate::helpers::composite::get_composite_shape(rhs);
    let lhs_flat = crate::helpers::composite::flatten_composite(lhs);
    let rhs_flat = crate::helpers::composite::flatten_composite(rhs);
    let use_float = lhs_flat.iter().any(|v| matches!(v, Value::Float(_)))
        || rhs_flat.iter().any(|v| matches!(v, Value::Float(_)));
    let use_complex = lhs_flat.iter().any(|v| matches!(v, Value::Complex { .. }))
        || rhs_flat.iter().any(|v| matches!(v, Value::Complex { .. }));

    if let (Value::List(ld), Value::List(rd)) = (lhs, rhs) {
        if rhs_shape.len() == 1 {
            // Matrix @ vector or vector @ vector
            if lhs_shape.len() == 1 {
                // 1D @ 1D: dot product → scalar
                if use_complex {
                    return matmul_dot_complex(b, &ld.values, &rd.values);
                }
                return matmul_dot(b, &ld.values, &rd.values, use_float);
            }
            // 2D @ 1D: each row dot product with vector → 1D
            let mut results = Vec::new();
            for row in &ld.values {
                if let Value::List(row_data) | Value::Tuple(row_data) = row {
                    if use_complex {
                        results.push(matmul_dot_complex(b, &row_data.values, &rd.values));
                    } else {
                        results.push(matmul_dot(b, &row_data.values, &rd.values, use_float));
                    }
                }
            }
            let types = results.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() });
        }

        if lhs_shape.len() == 1 && rhs_shape.len() == 2 {
            // 1D @ 2D: numpy prepends a leading 1 to lhs, multiplies,
            // then drops the prepended axis. (8,) @ (8, 7) → (1, 8) @ (8, 7)
            // = (1, 7) → (7,). Compute one row of dot products against each
            // column of rhs.
            let n = rhs_shape[1];
            let k = lhs_shape[0]; // = rhs_shape[0]
            let mut row_vals = Vec::with_capacity(n);
            for j in 0..n {
                let col: Vec<Value> = (0..k).map(|kk| {
                    let rhs_row = match &rd.values[kk] {
                        Value::List(r) | Value::Tuple(r) => &r.values,
                        _ => panic!("matmul: expected 2D array"),
                    };
                    rhs_row[j].clone()
                }).collect();
                let dot = if use_complex {
                    matmul_dot_complex(b, &ld.values, &col)
                } else {
                    matmul_dot(b, &ld.values, &col, use_float)
                };
                row_vals.push(dot);
            }
            let types = row_vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: types, values: row_vals, value_id: ValueId::next() });
        }

        if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
            // 2D @ 2D: full matrix multiply
            let m = lhs_shape[0]; // rows of lhs
            let k = lhs_shape[1]; // cols of lhs = rows of rhs
            let n = rhs_shape[1]; // cols of rhs

            let mut rows = Vec::new();
            for i in 0..m {
                let lhs_row = match &ld.values[i] {
                    Value::List(r) | Value::Tuple(r) => &r.values,
                    _ => panic!("matmul: expected 2D array"),
                };
                let mut row_vals = Vec::new();
                for j in 0..n {
                    if use_complex {
                        // Build a column-vector view by collecting rhs[*][j].
                        let col: Vec<Value> = (0..k).map(|kk| {
                            let rhs_row = match &rd.values[kk] {
                                Value::List(r) | Value::Tuple(r) => &r.values,
                                _ => panic!("matmul: expected 2D array"),
                            };
                            rhs_row[j].clone()
                        }).collect();
                        row_vals.push(matmul_dot_complex(b, lhs_row, &col));
                        continue;
                    }
                    // Compute dot product of lhs row i with rhs column j
                    let zero = if use_float {
                        b.ir_constant_float(0.0)
                    } else {
                        b.ir_constant_int(0)
                    };
                    let mut acc = zero;
                    for kk in 0..k {
                        let rhs_row = match &rd.values[kk] {
                            Value::List(r) | Value::Tuple(r) => &r.values,
                            _ => panic!("matmul: expected 2D array"),
                        };
                        let prod = if use_float {
                            let a = crate::helpers::value_ops::ensure_float(b, &lhs_row[kk]);
                            let b_val = crate::helpers::value_ops::ensure_float(b, &rhs_row[j]);
                            b.ir_mul_f(&a, &b_val)
                        } else {
                            b.ir_mul_i(&lhs_row[kk], &rhs_row[j])
                        };
                        acc = if use_float {
                            b.ir_add_f(&acc, &prod)
                        } else {
                            b.ir_add_i(&acc, &prod)
                        };
                    }
                    row_vals.push(acc);
                }
                let rtypes = row_vals.iter().map(|v| v.zinnia_type()).collect();
                rows.push(Value::List(CompositeData { elements_type: rtypes, values: row_vals, value_id: ValueId::next() }));
            }
            let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData { elements_type: row_types, values: rows, value_id: ValueId::next() });
        }
    }

    // Fallback: scalar multiply
    crate::helpers::value_ops::apply_binary_op(b, "mul", lhs, rhs)
}

/// Complex dot product: Σ aᵢ * bᵢ over Complex operands using
/// apply_binary_op so component-wise math goes through the existing
/// complex-arithmetic dispatch.
pub fn matmul_dot_complex(b: &mut IRBuilder, a: &[Value], bv: &[Value]) -> Value {
    let zero_re = b.ir_constant_float(0.0);
    let zero_im = b.ir_constant_float(0.0);
    let r = match zero_re {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let i = match zero_im {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let mut acc = Value::Complex { real: r, imag: i };
    let n = a.len().min(bv.len());
    for k in 0..n {
        let prod = crate::helpers::value_ops::apply_binary_op(b, "mul", &a[k], &bv[k]);
        acc = crate::helpers::value_ops::apply_binary_op(b, "add", &acc, &prod);
    }
    acc
}

/// Dot product helper for matmul.
pub fn matmul_dot(b: &mut IRBuilder, a: &[Value], bv: &[Value], use_float: bool) -> Value {
    let zero = if use_float {
        b.ir_constant_float(0.0)
    } else {
        b.ir_constant_int(0)
    };
    let mut acc = zero;
    for (x, y) in a.iter().zip(bv.iter()) {
        let prod = if use_float {
            let xf = crate::helpers::value_ops::ensure_float(b, x);
            let yf = crate::helpers::value_ops::ensure_float(b, y);
            b.ir_mul_f(&xf, &yf)
        } else {
            b.ir_mul_i(x, y)
        };
        acc = if use_float {
            b.ir_add_f(&acc, &prod)
        } else {
            b.ir_add_i(&acc, &prod)
        };
    }
    acc
}

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

/// Recursive concatenation along an arbitrary axis. At axis 0 we splice the
/// outer lists; at axis k > 0 we recurse into each outer position with
/// axis k − 1. All input arrays must have matching shapes except along the
/// concatenation axis (caller is expected to validate).
fn concat_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let mut all_values = Vec::new();
        let mut all_types = Vec::new();
        for arr in arrays {
            match arr {
                Value::List(d) | Value::Tuple(d) => {
                    all_values.extend(d.values.clone());
                    all_types.extend(d.elements_type.clone());
                }
                v => {
                    all_values.push(v.clone());
                    all_types.push(v.zinnia_type());
                }
            }
        }
        return Value::List(CompositeData { elements_type: all_types, values: all_values, value_id: ValueId::next() });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("concatenate: cannot apply axis > 0 to a 0-D value"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("concatenate: arrays must have matching ranks"),
            })
            .collect();
        rows.push(concat_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() })
}

pub fn np_concatenate(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len();
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array with {} dimensions",
            raw_axis, ndim
        );
    }

    let out = concat_recursive(&data.values, resolved as usize);
    let input_vids: Vec<ValueId> = data.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == data.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// Recursive stack along an arbitrary new axis. At axis 0 we wrap the input
/// arrays as the outer list directly. At axis k > 0 we walk the *first* axis
/// of the input arrays (which all have matching shapes) and recurse with
/// axis k − 1. The recursion bottoms out either in axis-0 wrap or in scalars.
fn stack_recursive(arrays: &[Value], axis: usize) -> Value {
    if axis == 0 {
        let types = arrays.iter().map(|v| v.zinnia_type()).collect();
        return Value::List(CompositeData {
            elements_type: types,
            values: arrays.to_vec(),
        
            value_id: ValueId::next(),
        });
    }
    let first = match &arrays[0] {
        Value::List(d) | Value::Tuple(d) => d,
        _ => panic!("stack: cannot stack at axis > input rank"),
    };
    let outer_len = first.values.len();
    let mut rows = Vec::with_capacity(outer_len);
    for i in 0..outer_len {
        let inner: Vec<Value> = arrays
            .iter()
            .map(|a| match a {
                Value::List(d) | Value::Tuple(d) => d.values[i].clone(),
                _ => panic!("stack: all input arrays must have the same rank"),
            })
            .collect();
        rows.push(stack_recursive(&inner, axis - 1));
    }
    let types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData { elements_type: types, values: rows, value_id: ValueId::next() })
}

pub fn np_stack(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let raw_axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);

    let data = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    if data.values.is_empty() {
        return Value::None;
    }

    // Stack inserts a new axis, so the result rank is input_rank + 1.
    let ndim = crate::helpers::composite::get_composite_shape(&data.values[0]).len() + 1;
    let resolved = if raw_axis < 0 { ndim as i64 + raw_axis } else { raw_axis };
    if resolved < 0 || resolved >= ndim as i64 {
        panic!(
            "axis {} is out of bounds for array of dimension {}",
            raw_axis,
            ndim - 1
        );
    }

    let out = stack_recursive(&data.values, resolved as usize);
    let input_vids: Vec<ValueId> = data.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == data.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

pub fn build_ndarray_from_flat(b: &mut IRBuilder, values: Vec<Value>, types: Vec<ZinniaType>, shape: &[usize]) -> Value {
    if shape.len() == 1 {
        Value::List(CompositeData { elements_type: types, values, value_id: ValueId::next() })
    } else {
        // Build nested structure
        let inner_size: usize = shape[1..].iter().product();
        let mut rows = Vec::new();
        for chunk in values.chunks(inner_size) {
            let chunk_types = chunk.iter().map(|v| v.zinnia_type()).collect();
            rows.push(build_ndarray_from_flat(b, chunk.to_vec(), chunk_types, &shape[1..]));
        }
        let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: row_types, values: rows, value_id: ValueId::next() })
    }
}

// ── NDArray helpers ───────────────────────────────────────────────

/// NDArray reshape: flatten, then rebuild with new shape.
pub fn ndarray_reshape(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let flat = crate::helpers::composite::flatten_composite(val);
    let total = flat.len();

    // Parse new shape from args — single tuple arg or multiple int args
    let new_shape: Vec<usize> = if args.len() == 1 {
        match &args[0] {
            Value::Tuple(data) | Value::List(data) => {
                data.values.iter().map(|v| {
                    let n = require_provable_static_int(b, v, SiteKind::ReshapeDim);
                    n as usize
                }).collect()
            }
            Value::Integer(_) => {
                let n = require_provable_static_int(b, &args[0], SiteKind::ReshapeDim);
                vec![n as usize]
            }
            _ => panic!("reshape: invalid shape argument"),
        }
    } else {
        args.iter().map(|v| {
            let n = require_provable_static_int(b, v, SiteKind::ReshapeDim);
            n as usize
        }).collect()
    };

    // Handle -1 (infer one dimension)
    let neg_count = new_shape.iter().filter(|&&s| s == usize::MAX).count();
    let final_shape: Vec<usize> = if neg_count == 1 {
        let known_product: usize = new_shape.iter().filter(|&&s| s != usize::MAX).product();
        assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
        new_shape.iter().map(|&s| if s == usize::MAX { total / known_product } else { s }).collect()
    } else {
        // Also handle -1 encoded as a large number from i64 cast
        let has_neg = args.iter().any(|a| a.int_val() == Some(-1));
        if has_neg {
            let known: Vec<usize> = new_shape.iter().copied().collect();
            let known_product: usize = known.iter().filter(|&&s| s < usize::MAX / 2).product();
            assert!(known_product > 0 && total % known_product == 0, "reshape: cannot infer dimension");
            known.iter().map(|&s| if s >= usize::MAX / 2 { total / known_product } else { s }).collect()
        } else {
            new_shape
        }
    };

    let shape_product: usize = final_shape.iter().product();
    assert_eq!(total, shape_product, "reshape: total size mismatch ({} vs {})", total, shape_product);

    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    build_ndarray_from_flat(b, flat, types, &final_shape)
}

/// NDArray moveaxis: reorder axes by moving source axis to destination.
pub fn ndarray_moveaxis(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "moveaxis: requires source and destination arguments");

    let src = {
        let s: i64 = require_provable_static_int(b, &args[0], SiteKind::Axis);
        if s < 0 { (ndim as i64 + s) as usize } else { s as usize }
    };
    let dst = {
        let d: i64 = require_provable_static_int(b, &args[1], SiteKind::Axis);
        if d < 0 { (ndim as i64 + d) as usize } else { d as usize }
    };
    assert!(src < ndim && dst < ndim, "moveaxis: axis out of bounds");

    // Build permutation: remove src, insert at dst
    let mut order: Vec<usize> = (0..ndim).filter(|&i| i != src).collect();
    order.insert(dst, src);

    let axes_val: Vec<Value> = order.iter()
        .map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None)))
        .collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; order.len()],
        values: axes_val,
    
        value_id: ValueId::next(),
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// NDArray repeat: repeat array elements along an axis.
pub fn ndarray_repeat(b: &mut IRBuilder, val: &Value, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let repeats_arg = args.first()
        .expect("repeat: repeats argument required");
    let axis = kwargs.get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    // Bounded-admission fast path: 1-D `val` with a scalar bounded `k` and
    // no axis (or axis=0) routes through zkRAM per-cell symbolic writes.
    // The output buffer has `len_arr * k_max` slots; slot `i` gets
    // `flat_src[(i / k) mod len_arr]`. Runtime mask via `runtime_length =
    // len_arr * k`. Multi-D input, tuple repeats, or axis != 0 fall
    // through to the static path.
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    if src_shape.len() == 1
        && matches!(repeats_arg, Value::Integer(_))
        && (axis.is_none() || axis == Some(0))
    {
        match resolve_int_or_bounded(b, repeats_arg, SiteKind::RepeatCount, None) {
            BoundedInt::Bounded { max, .. } => {
                let k_max = max.max(0) as usize;
                let flat_src = crate::helpers::composite::flatten_composite(val);
                let len_arr = flat_src.len();
                let dtype = match flat_src.first() {
                    Some(Value::Float(_)) => crate::types::NumberType::Float,
                    _ => crate::types::NumberType::Integer,
                };
                use crate::ops::dyn_ndarray::{
                    constructors::dyn_from_values_with_active, value_to_scalar_i64,
                };

                // Materialize source into a read-only segment for symbolic
                // reads. Length-zero input: nothing to repeat.
                if len_arr == 0 || k_max == 0 {
                    let runtime_length = b.ir_constant_int(0);
                    return dyn_from_values_with_active(
                        b,
                        Vec::new(),
                        runtime_length,
                        dtype,
                    );
                }
                let src_scalars: Vec<crate::types::ScalarValue<i64>> = flat_src
                    .iter()
                    .map(value_to_scalar_i64)
                    .collect();
                let src_seg = crate::helpers::segment::alloc_and_write(b, &src_scalars, dtype);

                // Output buffer pre-filled with defaults; per-cell symbolic
                // writes overwrite the active region. Slots beyond
                // runtime_length are never read by the subscript machinery.
                let default_sv = value_to_scalar_i64(&match dtype {
                    crate::types::NumberType::Float => b.ir_constant_float(0.0),
                    _ => b.ir_constant_int(0),
                });
                let init = vec![default_sv; len_arr * k_max];
                let out_seg = crate::helpers::segment::alloc_and_write(b, &init, dtype);

                let len_arr_const = b.ir_constant_int(len_arr as i64);
                let len_vid = len_arr_const.value_id().unwrap();
                for i in 0..(len_arr * k_max) {
                    let i_const = b.ir_constant_int(i as i64);
                    let arr_idx = b.ir_div_i(&i_const, repeats_arg);
                    let arr_idx_mod = b.ir_mod_i(&arr_idx, &len_arr_const);
                    let src_val = b.ir_read_memory(src_seg, &arr_idx_mod);
                    b.ir_write_memory(out_seg, &i_const, &src_val);
                }
                let runtime_length = b.ir_mul_i(&len_arr_const, repeats_arg);
                let runtime_length_sv = value_to_scalar_i64(&runtime_length);
                let runtime_length_vid = runtime_length_sv.value_id;

                // Fact: runtime_length == len_arr * k.
                let k_vid = repeats_arg.value_id().unwrap();
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                formals.insert("k".to_string(), k_vid);
                b.fire_contract("dyn_repeat", runtime_length_vid, &formals);

                let logical_shape = vec![len_arr * k_max];
                let envelope =
                    crate::types::Envelope::from_static_shape(&mut b.dim_table, &logical_shape);
                let result = Value::DynamicNDArray(crate::types::DynamicNDArrayData {
                    envelope,
                    dtype,
                    segment_id: out_seg,
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
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
            BoundedInt::Static(_) | BoundedInt::Neither => {
                // Fall through to the static path below.
            }
        }
    }

    let repeats: i64 = require_provable_static_int(b, repeats_arg, SiteKind::RepeatCount);

    if let Some(ax) = axis {
        // Repeat along specific axis
        let shape = crate::helpers::composite::get_composite_shape(val);
        let ndim = shape.len();
        let ax = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };

        if ax == 0 {
            if let Value::List(data) | Value::Tuple(data) = val {
                let mut new_vals = Vec::new();
                for v in &data.values {
                    for _ in 0..repeats {
                        new_vals.push(v.clone());
                    }
                }
                let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                let result = Value::List(CompositeData { elements_type: types, values: new_vals, value_id: ValueId::next() });
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
        }
        // For other axes, transpose so target axis is first, repeat, transpose back
        let mut fwd: Vec<usize> = (0..ndim).collect();
        fwd.swap(0, ax);
        let fwd_vals: Vec<Value> = fwd.iter().map(|&a| Value::Integer(crate::types::ScalarValue::new(Some(a as i64), None))).collect();
        let fwd_tuple = Value::Tuple(CompositeData { elements_type: vec![ZinniaType::Integer; ndim], values: fwd_vals, value_id: ValueId::next() });
        let transposed = crate::helpers::ndarray::ndarray_transpose(b, val, &[fwd_tuple.clone()]);
        let repeated = ndarray_repeat(b, &transposed, args, &HashMap::new());
        crate::helpers::ndarray::ndarray_transpose(b, &repeated, &[fwd_tuple])
    } else {
        // No axis: flatten, then repeat each element
        let flat = crate::helpers::composite::flatten_composite(val);
        let mut new_vals = Vec::new();
        for v in &flat {
            for _ in 0..repeats {
                new_vals.push(v.clone());
            }
        }
        let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
        let result = Value::List(CompositeData { elements_type: types, values: new_vals, value_id: ValueId::next() });
        if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
            crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
        }
        result
    }
}

/// NDArray filter: select elements where mask is true.
pub fn ndarray_filter(_b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let mask = args.first().expect("filter: requires a mask argument");
    let elements = crate::helpers::composite::flatten_composite(val);
    let mask_elements = crate::helpers::composite::flatten_composite(mask);
    assert_eq!(elements.len(), mask_elements.len(), "filter: array and mask must have same size");

    // For static arrays, we can build a filtered result at compile time
    // by using select chains. The result length depends on the mask values.
    // If mask values are all statically known, produce a fixed-size result.
    let mut static_result = Vec::new();
    let mut all_static = true;
    for (elem, m) in elements.iter().zip(mask_elements.iter()) {
        match m.int_val().or_else(|| if matches!(m, Value::Boolean(bv) if bv.static_val == Some(true)) { Some(1) } else if matches!(m, Value::Boolean(bv) if bv.static_val == Some(false)) { Some(0) } else { None }) {
            Some(v) if v != 0 => static_result.push(elem.clone()),
            Some(_) => {} // masked out
            None => { all_static = false; break; }
        }
    }

    if all_static {
        let types = static_result.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type: types, values: static_result, value_id: ValueId::next() })
    } else {
        panic!("filter: dynamic masks require DynamicNDArray (not yet supported in Rust backend)");
    }
}

pub fn ndarray_argmax_argmin_with_axis(b: &mut IRBuilder, val: &Value, axis: i64, is_max: bool) -> Value {
    if let Value::List(outer) | Value::Tuple(outer) = val {
        let ndim = crate::helpers::composite::get_composite_shape(val).len();
        let axis = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };

        if axis == 0 {
            // argmax along axis 0: for each column, find row with max/min
            if let Some(Value::List(first_row) | Value::Tuple(first_row)) = outer.values.first() {
                let ncols = first_row.values.len();
                let mut results = Vec::new();
                for col in 0..ncols {
                    let mut best_idx = b.ir_constant_int(0);
                    let mut best_val_opt: Option<Value> = None;
                    for (row_idx, row) in outer.values.iter().enumerate() {
                        if let Value::List(rd) | Value::Tuple(rd) = row {
                            if col < rd.values.len() {
                                if let Some(ref best_val) = best_val_opt {
                                    let cond = if is_max {
                                        b.ir_greater_than_i(&rd.values[col], best_val)
                                    } else {
                                        b.ir_less_than_i(&rd.values[col], best_val)
                                    };
                                    let idx_val = b.ir_constant_int(row_idx as i64);
                                    best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
                                    best_val_opt = Some(b.ir_select_i(&cond, &rd.values[col], best_val));
                                } else {
                                    best_val_opt = Some(rd.values[col].clone());
                                }
                            }
                        }
                    }
                    results.push(best_idx);
                }
                let types = vec![ZinniaType::Integer; results.len()];
                return Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() });
            }
        } else if axis == 1 {
            // argmax along axis 1: for each row, find column index of max/min
            let mut results = Vec::new();
            for row in &outer.values {
                results.push(crate::helpers::ndarray::ndarray_argmax_argmin(b, row, &[], is_max));
            }
            let types = vec![ZinniaType::Integer; results.len()];
            return Value::List(CompositeData { elements_type: types, values: results, value_id: ValueId::next() });
        }
    }
    crate::helpers::ndarray::ndarray_argmax_argmin(b, val, &[], is_max)
}


pub fn ndarray_shape(val: &Value) -> Value {
    // Return the shape as a tuple of constants
    match val {
        Value::List(data) => {
            // For a list, shape is (len,)
            let len_val = Value::Integer(crate::types::ScalarValue::new(Some(data.values.len() as i64), None));
            Value::Tuple(CompositeData {
                elements_type: vec![ZinniaType::Integer],
                values: vec![len_val],
            
                value_id: ValueId::next(),
            })
        }
        Value::NDArray(nd) => {
            let shape_vals: Vec<Value> = nd.shape.iter()
                .map(|&s| Value::Integer(crate::types::ScalarValue::new(Some(s as i64), None)))
                .collect();
            let types = shape_vals.iter().map(|_| ZinniaType::Integer).collect();
            Value::Tuple(CompositeData {
                elements_type: types,
                values: shape_vals,
            
                value_id: ValueId::next(),
            })
        }
        _ => Value::None,
    }
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
// Shape-manipulation helpers (swapaxes / flip / squeeze / expand_dims /
// broadcast_to / atleast_Nd / tile / vstack / hstack / dstack /
// column_stack / row_stack)
// ────────────────────────────────────────────────────────────────────────

// `resolve_axis` lives in `helpers::shape_arith`. Re-export under the local
// path so the rest of this file can call it unqualified, exactly as before.
use crate::helpers::shape_arith::resolve_axis;

/// Build a constant Integer Value from a usize.
fn const_int(b: &mut IRBuilder, n: usize) -> Value {
    b.ir_constant_int(n as i64)
}

/// `np.swapaxes(arr, a1, a2)` — swap two axes.
pub fn ndarray_swapaxes(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    assert!(args.len() >= 2, "swapaxes: requires two axis arguments");
    let a1 = resolve_axis(
        require_provable_static_int(b, &args[0], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let a2 = resolve_axis(
        require_provable_static_int(b, &args[1], SiteKind::Axis),
        ndim,
        "swapaxes",
    );
    let mut order: Vec<usize> = (0..ndim).collect();
    order.swap(a1, a2);
    let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
    let axes_tuple = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; ndim],
        values: axes_vals,
    
        value_id: ValueId::next(),
    });
    crate::helpers::ndarray::ndarray_transpose(b, val, &[axes_tuple])
}

/// Reverse `val`'s elements along axis `axis`. Other axes keep order.
fn flip_along(val: &Value, axis: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values.iter().rev().cloned().collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                
                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> =
                    d.values.iter().map(|v| flip_along(v, axis - 1)).collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                
                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    }
}

/// `np.flip(arr, axis=None)` — reverse along the given axis (or all axes
/// when no axis is specified).
pub fn np_flip(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("flip: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));
    let axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => (0..ndim).collect(),
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                resolve_axis(n, ndim, "flip")
            })
            .collect(),
        Some(a) => {
            let n: i64 = require_provable_static_int(b, a, SiteKind::Axis);
            vec![resolve_axis(n, ndim, "flip")]
        }
    };
    let mut out = val.clone();
    for ax in axes {
        out = flip_along(&out, ax);
    }
    out
}

/// `np.flipud(arr)` — flip along axis 0.
pub fn np_flipud(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("flipud: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.is_empty() {
        panic!("flipud: input must be at least 1-D");
    }
    flip_along(val, 0)
}

/// `np.fliplr(arr)` — flip along axis 1 (requires ndim ≥ 2).
pub fn np_fliplr(_b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("fliplr: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("fliplr: input must be at least 2-D");
    }
    flip_along(val, 1)
}

/// `np.rot90(arr, k=1, axes=(0, 1))` — rotate 90° counter-clockwise k times
/// in the plane spanned by `axes`. Each k=1 rotation is `flip(axes[1])`
/// then `swapaxes(axes[0], axes[1])`, matching NumPy's reference.
pub fn np_rot90(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("rot90: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim < 2 {
        panic!("rot90: input must be at least 2-D");
    }
    let k = kwargs
        .get("k")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(1);
    let axes_arg = kwargs.get("axes").or_else(|| args.get(2));
    let (a0, a1) = match axes_arg {
        Some(Value::Tuple(d)) | Some(Value::List(d)) if d.values.len() == 2 => {
            let a = resolve_axis(d.values[0].int_val().unwrap_or(0), ndim, "rot90");
            let bb = resolve_axis(d.values[1].int_val().unwrap_or(1), ndim, "rot90");
            (a, bb)
        }
        _ => (0usize, 1usize),
    };
    if a0 == a1 {
        panic!("rot90: axes must be different");
    }
    let k = ((k % 4) + 4) % 4;
    let mut out = val.clone();
    for _ in 0..k {
        out = flip_along(&out, a1);
        let mut order: Vec<usize> = (0..ndim).collect();
        order.swap(a0, a1);
        let axes_vals: Vec<Value> = order.iter().map(|&a| const_int(b, a)).collect();
        let axes_tuple = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer; ndim],
            values: axes_vals,
        
            value_id: ValueId::next(),
        });
        out = crate::helpers::ndarray::ndarray_transpose(b, &out, &[axes_tuple]);
    }
    out
}

/// `np.squeeze(arr, axis=None)` — drop axes of length 1.
pub fn np_squeeze(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("squeeze: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_val = kwargs.get("axis").or_else(|| args.get(1));

    let target_axes: Vec<usize> = match axis_val {
        Some(Value::None) | None => shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 1 { Some(i) } else { None })
            .collect(),
        Some(Value::Tuple(d)) | Some(Value::List(d)) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::Axis);
                resolve_axis(n, ndim, "squeeze")
            })
            .collect(),
        Some(a) => {
            let n: i64 = require_provable_static_int(b, a, SiteKind::Axis);
            vec![resolve_axis(n, ndim, "squeeze")]
        }
    };
    for &ax in &target_axes {
        if shape[ax] != 1 {
            panic!(
                "squeeze: cannot select an axis to squeeze out which has size not equal to one (axis {})",
                ax
            );
        }
    }
    if target_axes.is_empty() {
        return val.clone();
    }
    let new_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if target_axes.contains(&i) { None } else { Some(d) })
        .collect();
    let flat = crate::helpers::composite::flatten_composite(val);
    if new_shape.is_empty() {
        let out = flat.into_iter().next().unwrap_or(Value::None);
        if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
            crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
        }
        return out;
    }
    let types = flat.iter().map(|v| v.zinnia_type()).collect();
    let out = crate::helpers::composite::build_nested_value(flat, types, &new_shape);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.expand_dims(arr, axis)` — insert a new axis of length 1 at `axis`.
pub fn np_expand_dims(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("expand_dims: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    let axis_arg = args
        .get(1)
        .expect("expand_dims: axis argument required");
    let axis: i64 = require_provable_static_int(b, axis_arg, SiteKind::NewAxisPosition);
    let new_ndim = ndim + 1;
    let resolved = if axis < 0 { new_ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved >= new_ndim as i64 {
        panic!(
            "expand_dims: axis {} is out of bounds for array of rank {}",
            axis, new_ndim
        );
    }
    let pos = resolved as usize;
    fn insert_at(val: &Value, pos: usize) -> Value {
        if pos == 0 {
            Value::List(CompositeData {
                elements_type: vec![val.zinnia_type()],
                values: vec![val.clone()],

                value_id: ValueId::next(),
            })
        } else {
            match val {
                Value::List(d) | Value::Tuple(d) => {
                    let new_vals: Vec<Value> =
                        d.values.iter().map(|v| insert_at(v, pos - 1)).collect();
                    let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                    Value::List(CompositeData {
                        elements_type: new_types,
                        values: new_vals,

                        value_id: ValueId::next(),
                    })
                }
                _ => val.clone(),
            }
        }
    }
    let out = insert_at(val, pos);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.broadcast_to(arr, shape)` — materialize the broadcast to a target
/// shape. Thin wrapper around the broadcasting helper.
pub fn np_broadcast_to(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("broadcast_to: requires an array argument");
    let shape_arg = args.get(1).expect("broadcast_to: requires a shape argument");
    let target: Vec<usize> = match shape_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::ShapeAxis(i));
                n as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n: i64 = require_provable_static_int(b, shape_arg, SiteKind::ShapeAxis(0));
            vec![n as usize]
        }
        _ => panic!("broadcast_to: invalid shape argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    match crate::helpers::broadcast::broadcast_shapes(&src_shape, &target) {
        Some(s) if s == target => {}
        _ => panic!(
            "broadcast_to: shape {:?} cannot be broadcast to {:?}",
            src_shape, target
        ),
    }
    // broadcast_to is sound for relay: it replicates existing cells, so every
    // new element equals an existing one — `forall_eq_const(in, k)` ⇒
    // `forall_eq_const(out, k)`.
    let out = crate::helpers::broadcast::materialize_to_shape(val, &target);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

/// `np.atleast_1d/2d/3d(arr)` — prepend unit axes until rank ≥ n.
pub fn np_atleast_nd(_b: &mut IRBuilder, args: &[Value], n: usize) -> Value {
    let val = args.first().expect("atleast_Nd: requires an array argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() >= n {
        return val.clone();
    }
    let mut out = val.clone();
    for _ in 0..(n - shape.len()) {
        out = Value::List(CompositeData {
            elements_type: vec![out.zinnia_type()],
            values: vec![out],
        
            value_id: ValueId::next(),
        });
    }
    out
}

/// `np.tile(arr, reps)` — repeat `arr` according to `reps`.
pub fn np_tile(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("tile: requires an array argument");
    let reps_arg = args.get(1).expect("tile: requires a reps argument");

    // Bounded-admission fast path: 1-D `arr` with a scalar bounded `reps`
    // promotes to a `DynamicNDArray` via the natural-pad approach.
    // `tile([a,b,c], k)` for `k ∈ [0, k_max]` is a prefix of the fully
    // expanded `tile([a,b,c], k_max)`; padding to `k_max` and truncating
    // via `runtime_length = len_arr * k` gives the right user-visible
    // output. (Multi-D bounded tile is out of scope.)
    if matches!(reps_arg, Value::Integer(_))
        && crate::helpers::composite::get_composite_shape(val).len() == 1
    {
        match resolve_int_or_bounded(b, reps_arg, SiteKind::RepeatCount, None) {
            BoundedInt::Bounded { max, .. } => {
                let k_max = max.max(0) as usize;
                let flat_src = crate::helpers::composite::flatten_composite(val);
                let len_arr = flat_src.len();
                let dtype = match flat_src.first() {
                    Some(Value::Float(_)) => crate::types::NumberType::Float,
                    _ => crate::types::NumberType::Integer,
                };
                use crate::ops::dyn_ndarray::{
                    constructors::dyn_from_values_with_active, value_to_scalar_i64,
                };
                let mut values: Vec<crate::types::ScalarValue<i64>> =
                    Vec::with_capacity(len_arr * k_max);
                for _ in 0..k_max {
                    for v in &flat_src {
                        values.push(value_to_scalar_i64(v));
                    }
                }
                let len_const = b.ir_constant_int(len_arr as i64);
                let len_vid = len_const.value_id().unwrap();
                let runtime_length = b.ir_mul_i(&len_const, reps_arg);
                let result = dyn_from_values_with_active(b, values, runtime_length, dtype);
                // Fact: runtime_length == len_arr * k.
                let runtime_length_vid = match &result {
                    Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
                    _ => unreachable!(),
                };
                let k_vid = reps_arg.value_id().unwrap();
                let mut formals = std::collections::HashMap::new();
                formals.insert("len_arr".to_string(), len_vid);
                formals.insert("k".to_string(), k_vid);
                b.fire_contract("dyn_tile", runtime_length_vid, &formals);
                if let (Some(in_vid), Some(out_vid)) = (val.value_id(), result.value_id()) {
                    crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
                }
                return result;
            }
            BoundedInt::Static(_) | BoundedInt::Neither => {
                // Fall through to the static path below, which will
                // either succeed (Static) or panic with the standard
                // RepeatCount diagnostic (Neither).
            }
        }
    }

    let reps: Vec<usize> = match reps_arg {
        Value::Tuple(d) | Value::List(d) => d
            .values
            .iter()
            .map(|v| {
                let n: i64 = require_provable_static_int(b, v, SiteKind::RepeatCount);
                n.max(0) as usize
            })
            .collect(),
        Value::Integer(_) => {
            let n: i64 = require_provable_static_int(b, reps_arg, SiteKind::RepeatCount);
            vec![n.max(0) as usize]
        }
        _ => panic!("tile: invalid reps argument"),
    };
    let src_shape = crate::helpers::composite::get_composite_shape(val);
    let rank = src_shape.len().max(reps.len());
    let mut padded_shape = vec![1usize; rank - src_shape.len()];
    padded_shape.extend_from_slice(&src_shape);
    let mut padded_reps = vec![1usize; rank - reps.len()];
    padded_reps.extend_from_slice(&reps);

    // Promote val to padded rank by prepending unit axes if needed.
    let mut promoted = val.clone();
    for _ in 0..(rank - src_shape.len()) {
        promoted = Value::List(CompositeData {
            elements_type: vec![promoted.zinnia_type()],
            values: vec![promoted],
        
            value_id: ValueId::next(),
        });
    }

    let target_shape: Vec<usize> = padded_shape
        .iter()
        .zip(padded_reps.iter())
        .map(|(s, r)| s * r)
        .collect();

    let total: usize = target_shape.iter().product();
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * target_shape[i + 1];
    }
    let mut src_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * padded_shape[i + 1];
    }
    let flat_src = crate::helpers::composite::flatten_composite(&promoted);
    let mut out_flat: Vec<Value> = Vec::with_capacity(total);
    for out_idx in 0..total {
        let mut remainder = out_idx;
        let mut src_flat = 0usize;
        for d in 0..rank {
            let coord = remainder / out_strides[d];
            remainder %= out_strides[d];
            let src_coord = coord % padded_shape[d];
            src_flat += src_coord * src_strides[d];
        }
        out_flat.push(flat_src[src_flat].clone());
    }
    let types = out_flat.iter().map(|v| v.zinnia_type()).collect();
    let out = crate::helpers::composite::build_nested_value(out_flat, types, &target_shape);
    if let (Some(in_vid), Some(out_vid)) = (val.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

// ── stack convenience wrappers ─────────────────────────────────────────

/// Promote a 1-D array to a 2-D row (`(N,)` → `(1, N)`); leave higher-rank
/// arrays untouched. Used by vstack/row_stack.
fn promote_to_row(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        Value::List(CompositeData {
            elements_type: vec![val.zinnia_type()],
            values: vec![val.clone()],
        
            value_id: ValueId::next(),
        })
    } else {
        val.clone()
    }
}

/// Promote a 1-D array to a 2-D column (`(N,)` → `(N, 1)`).
fn promote_to_column(val: &Value) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() == 1 {
        if let Value::List(d) | Value::Tuple(d) = val {
            let new_vals: Vec<Value> = d
                .values
                .iter()
                .map(|v| {
                    Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],
                    
                        value_id: ValueId::next(),
                    })
                })
                .collect();
            let types = new_vals.iter().map(|v| v.zinnia_type()).collect();
            return Value::List(CompositeData {
                elements_type: types,
                values: new_vals,
            
                value_id: ValueId::next(),
            });
        }
    }
    val.clone()
}

/// `np.vstack(arrays)` — stack along axis 0, promoting 1-D inputs to rows.
pub fn np_vstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_row).collect();
    let out = concat_recursive(&promoted, 0);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.hstack(arrays)` — concatenate along axis 1 for ≥2-D inputs, or along
/// axis 0 for 1-D inputs (NumPy convention).
pub fn np_hstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let any_multi = arrays
        .values
        .iter()
        .any(|v| crate::helpers::composite::get_composite_shape(v).len() >= 2);
    let axis = if any_multi { 1 } else { 0 };
    let out = concat_recursive(&arrays.values, axis);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.dstack(arrays)` — stack along axis 2, promoting lower-rank inputs.
pub fn np_dstack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let raw_values = arrays.values.clone();
    let promoted: Vec<Value> = raw_values
        .iter()
        .map(|v| {
            let shape = crate::helpers::composite::get_composite_shape(v);
            match shape.len() {
                1 => {
                    // (N,) -> (1, N) -> (1, N, 1)
                    let row = Value::List(CompositeData {
                        elements_type: vec![v.zinnia_type()],
                        values: vec![v.clone()],

                        value_id: ValueId::next(),
                    });
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[row, two])
                }
                2 => {
                    let two = b.ir_constant_int(2);
                    np_expand_dims(b, &[v.clone(), two])
                }
                _ => v.clone(),
            }
        })
        .collect();
    let out = concat_recursive(&promoted, 2);
    let input_vids: Vec<ValueId> = raw_values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == raw_values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.column_stack(arrays)` — 1-D arrays become columns of a 2-D output.
pub fn np_column_stack(b: &mut IRBuilder, args: &[Value]) -> Value {
    let arrays = match args.first() {
        Some(Value::List(d)) | Some(Value::Tuple(d)) => d,
        _ => return Value::None,
    };
    let promoted: Vec<Value> = arrays.values.iter().map(promote_to_column).collect();
    let out = concat_recursive(&promoted, 1);
    let input_vids: Vec<ValueId> = arrays.values.iter().filter_map(|v| v.value_id()).collect();
    if let Some(out_vid) = out.value_id() {
        if input_vids.len() == arrays.values.len() {
            crate::optim::resolver::relay_forall_eq_const_from_all_inputs(b, &input_vids, out_vid);
        }
    }
    out
}

/// `np.row_stack(arrays)` — alias of vstack.
pub fn np_row_stack(b: &mut IRBuilder, args: &[Value]) -> Value {
    np_vstack(b, args)
}

// ────────────────────────────────────────────────────────────────────────
// Element-wise math: round / floor / ceil / trunc / reciprocal / where /
// clip. None of these have a dedicated IR primitive yet, so they are
// expressed in terms of existing ops (floor_div, sign, select, etc).
// ────────────────────────────────────────────────────────────────────────

/// Recursively apply `scalar` to every leaf in `val`. Used by all the
/// element-wise wrappers below — keeps the leaf-walking boilerplate in
/// one place.
fn vectorize_unary<F: FnMut(&mut IRBuilder, &Value) -> Value>(
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
        let mean_expanded = np_expand_dims(b, &[mean.clone(), axis_const]);
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
        let types: Vec<crate::types::ZinniaType> = flat.iter().map(|v| v.zinnia_type()).collect();
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

// ────────────────────────────────────────────────────────────────────────
// Splitting family (split / array_split / hsplit / vsplit / dsplit)
// ────────────────────────────────────────────────────────────────────────

/// Take the slice `start..stop` of `val` along axis `axis`. Other axes are
/// kept intact. Used by all the splitting helpers below.
fn slice_along_axis(val: &Value, axis: usize, start: usize, stop: usize) -> Value {
    if axis == 0 {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d.values[start..stop].to_vec();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                
                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    } else {
        match val {
            Value::List(d) | Value::Tuple(d) => {
                let new_vals: Vec<Value> = d
                    .values
                    .iter()
                    .map(|v| slice_along_axis(v, axis - 1, start, stop))
                    .collect();
                let new_types = new_vals.iter().map(|v| v.zinnia_type()).collect();
                Value::List(CompositeData {
                    elements_type: new_types,
                    values: new_vals,
                
                    value_id: ValueId::next(),
                })
            }
            _ => val.clone(),
        }
    }
}

/// Compute the section boundaries (cumulative sizes) for `np.split` and
/// `np.array_split`. For `array_split`, when `n` does not evenly divide
/// `length`, the first `length % n` sections get one extra element each
/// (matching NumPy).
fn compute_split_boundaries(
    b: &mut IRBuilder,
    length: usize,
    sections: &Value,
    allow_uneven: bool,
) -> Vec<(usize, usize)> {
    match sections {
        Value::Integer(_) => {
            let n = require_provable_static_int(b, sections, SiteKind::SplitSections);
            let n = n as usize;
            if n == 0 {
                panic!("split: number of sections must be > 0");
            }
            if !allow_uneven && length % n != 0 {
                panic!(
                    "split: array of length {} cannot be split into {} equal sections",
                    length, n
                );
            }
            let base = length / n;
            let extras = length % n;
            let mut out = Vec::with_capacity(n);
            let mut cursor = 0usize;
            for i in 0..n {
                let sz = base + if i < extras { 1 } else { 0 };
                out.push((cursor, cursor + sz));
                cursor += sz;
            }
            out
        }
        Value::List(d) | Value::Tuple(d) => {
            // Index list: split *at* these indices.
            let mut indices: Vec<usize> = d
                .values
                .iter()
                .map(|v| {
                    let i = require_provable_static_int(b, v, SiteKind::SplitSections);
                    i.max(0).min(length as i64) as usize
                })
                .collect();
            indices.push(length);
            let mut out = Vec::with_capacity(indices.len());
            let mut prev = 0usize;
            for &i in &indices {
                out.push((prev, i.max(prev)));
                prev = i.max(prev);
            }
            out
        }
        _ => panic!("split: sections must be an int or a list of indices"),
    }
}

/// Shared body for `np.split` / `np.array_split` along an explicit axis.
fn split_impl(
    b: &mut IRBuilder,
    val: &Value,
    sections: &Value,
    axis: i64,
    allow_uneven: bool,
    op: &str,
) -> Value {
    let shape = crate::helpers::composite::get_composite_shape(val);
    let ndim = shape.len();
    if ndim == 0 {
        panic!("{}: cannot split a 0-D value", op);
    }
    let ax = resolve_axis(axis, ndim, op);
    let length = shape[ax];
    let bounds = compute_split_boundaries(b, length, sections, allow_uneven);
    let parts: Vec<Value> = bounds
        .into_iter()
        .map(|(s, e)| slice_along_axis(val, ax, s, e))
        .collect();
    let types = parts.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values: parts,
    
        value_id: ValueId::next(),
    })
}

/// `np.split(arr, sections, axis=0)` — equal-section split (errors if the
/// sections don't divide evenly).
pub fn np_split(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let val = args.first().expect("split: requires an array argument");
    let sections = args.get(1).expect("split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(b, val, sections, axis, false, "split")
}

/// `np.array_split(arr, sections, axis=0)` — like split but allows uneven
/// sections; the first `length % n` sections get one extra element.
pub fn np_array_split(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let val = args.first().expect("array_split: requires an array argument");
    let sections = args.get(1).expect("array_split: requires a sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(2))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    split_impl(b, val, sections, axis, true, "array_split")
}

/// `np.hsplit(arr, sections)` — split along axis 1 for ≥2-D, axis 0 for 1-D.
pub fn np_hsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("hsplit: requires an array argument");
    let sections = args.get(1).expect("hsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    let axis = if shape.len() >= 2 { 1 } else { 0 };
    split_impl(b, val, sections, axis, false, "hsplit")
}

/// `np.vsplit(arr, sections)` — split along axis 0 (requires ≥2-D).
pub fn np_vsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("vsplit: requires an array argument");
    let sections = args.get(1).expect("vsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 2 {
        panic!("vsplit: input must be at least 2-D");
    }
    split_impl(b, val, sections, 0, false, "vsplit")
}

/// `np.dsplit(arr, sections)` — split along axis 2 (requires ≥3-D).
pub fn np_dsplit(b: &mut IRBuilder, args: &[Value]) -> Value {
    let val = args.first().expect("dsplit: requires an array argument");
    let sections = args.get(1).expect("dsplit: requires a sections argument");
    let shape = crate::helpers::composite::get_composite_shape(val);
    if shape.len() < 3 {
        panic!("dsplit: input must be at least 3-D");
    }
    split_impl(b, val, sections, 2, false, "dsplit")
}

// ────────────────────────────────────────────────────────────────────────
// np.block — recursive nested concatenation
// ────────────────────────────────────────────────────────────────────────

/// Recursive worker for `np.block`. At each block level (going from the
/// outermost level inward), we recurse on each child with `block_depth − 1`,
/// then concat the results along the appropriate axis.
///
/// The axis follows NumPy's "negative axis from the result rank" rule: at
/// the outermost level we concat along axis `result_ndim − block_depth`; at
/// the innermost block level we concat along axis `result_ndim − 1`.
fn block_recursive(val: &Value, block_depth: usize, result_ndim: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }
    let children: Vec<Value> = match val {
        Value::List(d) | Value::Tuple(d) => d
            .values
            .iter()
            .map(|c| block_recursive(c, block_depth - 1, result_ndim))
            .collect(),
        _ => return val.clone(),
    };
    let axis = result_ndim - block_depth;
    concat_recursive(&children, axis)
}

/// Walk every leaf in the nested block structure at the given depth. Used to
/// validate that all leaves share a rank.
fn collect_block_leaves(val: &Value, block_depth: usize) -> Vec<Value> {
    if block_depth == 0 {
        return vec![val.clone()];
    }
    match val {
        Value::List(d) | Value::Tuple(d) => {
            let mut all = Vec::new();
            for child in &d.values {
                all.extend(collect_block_leaves(child, block_depth - 1));
            }
            all
        }
        _ => vec![val.clone()],
    }
}

/// `np.block(arrays, block_depth)` — recursive nested concatenation. The
/// block depth must be supplied by the caller (typically computed from the
/// AST nesting in `ir_gen/named_attr.rs`, since after a Python list literal
/// has been visited into a `Value::List` we can no longer distinguish "a
/// nested block of arrays" from "a single high-rank ndarray").
///
/// All leaf arrays must share the same rank, and that rank must be ≥
/// `block_depth`. NumPy auto-promotes mixed-rank leaves via `atleast_Nd`;
/// that case is currently out of scope and produces a hard error.
pub fn np_block_with_depth(val: &Value, block_depth: usize) -> Value {
    if block_depth == 0 {
        return val.clone();
    }

    let leaves = collect_block_leaves(val, block_depth);
    if leaves.is_empty() {
        return val.clone();
    }
    let first_rank = crate::helpers::composite::get_composite_shape(&leaves[0]).len();
    for leaf in &leaves {
        let r = crate::helpers::composite::get_composite_shape(leaf).len();
        if r != first_rank {
            panic!(
                "block: all leaf arrays must currently have the same rank \
                 (got {} and {}). Mixed-rank block (NumPy auto-promotes via \
                 atleast_Nd) is not yet supported on static ndarrays.",
                first_rank, r
            );
        }
    }
    if first_rank < block_depth {
        panic!(
            "block: leaf arrays of rank {} cannot be combined into a block \
             of nesting depth {}. Promote them with np.atleast_Nd first, or \
             use np.stack / np.concatenate directly.",
            first_rank, block_depth
        );
    }

    let result_ndim = first_rank.max(block_depth);
    block_recursive(val, block_depth, result_ndim)
}

// ────────────────────────────────────────────────────────────────────────
// Tests — bound-aware reshape / split chokepoints
// (compiler.bound-aware-reshape, compiler.bound-aware-split)
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
    use crate::optim::resolver::LayeredResolver;
    use crate::types::ScalarValue;

    /// Build a 1-D static-NDArray Value::List of `n` literal floats.
    fn static_floats(n: usize) -> Value {
        let values: Vec<Value> = (0..n)
            .map(|i| Value::Float(ScalarValue::constant(i as f64)))
            .collect();
        let types = vec![ZinniaType::Float; n];
        Value::List(CompositeData {
            elements_type: types,
            values,
        
            value_id: ValueId::next(),
        })
    }

    /// Plant `k * k == n_squared` and `k >= 0` so prove() pins k to the
    /// positive sqrt. Shape-matching scanner can't decompose Arith; prove
    /// can.
    fn plant_k_squared_eq(b: &mut IRBuilder, k_vid: crate::types::ValueId, n_squared: i64) {
        let k_sq = ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        };
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Eq,
                lhs: Box::new(k_sq),
                rhs: Box::new(ContractTerm::LitInt(n_squared)),
            },
        );
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
        );
    }

    #[test]
    fn reshape_admits_value_provable_via_prove() {
        // `arr.reshape(k, m)` where k = 4 follows from `k * k == 16` and
        // `k >= 0`. The scanner shape-matches only `Cmp(Value, LitInt)`;
        // the arithmetic shape requires prove(). Today's `require_static_int`
        // would reject this program; `resolve_int_or_bounded` admits it.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let arr = static_floats(8); // 1-D of length 8

        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        plant_k_squared_eq(&mut b, k_vid, 16); // k == 4
        let two = b.ir_constant_int(2);

        // reshape(arr, k, 2) → shape (4, 2).
        let out = ndarray_reshape(&mut b, &arr, &[k, two]);
        let shape = crate::helpers::composite::get_composite_shape(&out);
        assert_eq!(shape, vec![4, 2], "reshape did not produce the prove-derived shape");
    }

    #[test]
    fn split_admits_value_provable_via_prove() {
        // np.split(arr, k) with k == 2 (derived from `k * k == 4`).
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let arr = static_floats(8); // 1-D length 8, splits into 2 sections of 4

        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        plant_k_squared_eq(&mut b, k_vid, 4); // k == 2

        let kwargs = std::collections::HashMap::new();
        let out = np_split(&mut b, &[arr, k], &kwargs);
        // Result is Value::List of 2 sub-lists.
        if let Value::List(d) = &out {
            assert_eq!(d.values.len(), 2, "split did not yield 2 sections");
            // Each section has 4 elements.
            for sec in &d.values {
                let sh = crate::helpers::composite::get_composite_shape(sec);
                assert_eq!(sh, vec![4]);
            }
        } else {
            panic!("expected Value::List from np_split, got {:?}", out);
        }
    }

    #[test]
    #[should_panic(expected = "reshape target dimension must be a compile-time constant int")]
    fn reshape_still_rejects_when_no_facts() {
        // Without facts, an unconstrained k must still reject — the
        // bound-aware path returns Neither, falling through to the same
        // diagnostic as before.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let arr = static_floats(8);
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let two = b.ir_constant_int(2);
        let _ = ndarray_reshape(&mut b, &arr, &[k, two]);
    }

    #[test]
    #[should_panic(expected = "split sections must be a compile-time constant int")]
    fn split_still_rejects_when_no_facts() {
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
        let arr = static_floats(8);
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let kwargs = std::collections::HashMap::new();
        let _ = np_split(&mut b, &[arr, k], &kwargs);
    }

    #[test]
    fn np_fill_1d_admits_value_provable_via_prove_bounded() {
        // `np.zeros(k, dtype=int)` where `k ∈ [0, 10]` follows from
        // `k + k <= 20 ∧ k + k >= 0`. The shape-matching scanner only
        // decomposes `Cmp(Value, LitInt)`; the arithmetic bound requires
        // prove(). `resolve_int_or_bounded` admits this via the outward-
        // doubling probe and the constructor promotes to a 1-D
        // `DynamicNDArray` whose envelope's max_total is 10.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();

        let k_plus_k = ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        };
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(k_plus_k.clone()),
                rhs: Box::new(ContractTerm::LitInt(20)),
            },
        );
        b.facts.insert_for(
            k_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(k_plus_k),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
        );

        let kwargs = std::collections::HashMap::new();
        let out = np_fill(&mut b, &[k], &kwargs, 0);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                10,
                "expected prove-derived max_length=10 from `k + k <= 20`",
            );
        } else {
            panic!("expected Value::DynamicNDArray from np_fill on bounded k, got {:?}", out);
        }
    }

    /// Plant `n + n <= 20 ∧ n + n >= 0`, which prove() decomposes to
    /// `n ∈ [0, 10]`. The shape-matching scanner can't see through
    /// arithmetic; only `resolve_int_or_bounded`'s outward-doubling probe
    /// can. Used by the np_arange / np_tile bounded-admission tests.
    fn plant_bounded_zero_to_ten(b: &mut IRBuilder, n_vid: crate::types::ValueId) {
        let n_plus_n = ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        };
        b.facts.insert_for(
            n_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(n_plus_n.clone()),
                rhs: Box::new(ContractTerm::LitInt(20)),
            },
        );
        b.facts.insert_for(
            n_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(n_plus_n),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
        );
    }

    #[test]
    fn np_arange_admits_value_provable_via_prove_bounded() {
        // `np.arange(n)` where `n ∈ [0, 10]`. The bounded-admission path
        // builds a 1-D `DynamicNDArray` with envelope max_length=10 and
        // runtime_length aliasing the user's `n`. The buffer's prefix
        // values are `[0, 1, ..., 9]`; the tail beyond runtime_length is
        // masked by downstream subscript ops.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
        let n_vid = n.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, n_vid);

        let out = np_arange(&mut b, &[n.clone()]);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                10,
                "expected prove-derived max_length=10 from `n + n <= 20`",
            );
            assert_eq!(
                data.meta.runtime_length.value_id, n_vid,
                "runtime_length should alias n's value_id",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_arange on bounded n, got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_tile_admits_value_provable_via_prove_bounded() {
        // `np.tile(arr, k)` where `arr` is a 3-element static 1-D array
        // and `k ∈ [0, 10]`. The bounded-admission path natural-pads to
        // `k_max=10` (buffer length `3 * 10 = 30`) and computes
        // `runtime_length = 3 * k`.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let arr = static_floats(3);
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, k_vid);

        let out = np_tile(&mut b, &[arr, k.clone()]);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                30,
                "expected prove-derived max_length=3*10 from arr.len()=3 and `k <= 10`",
            );
            // runtime_length is the SSA value `3 * k`; we can't compare to
            // k_vid directly (it's a fresh MulI output), but it should have
            // a value_id and should not equal k_vid.
            assert_ne!(
                data.meta.runtime_length.value_id, k_vid,
                "runtime_length should be the `3 * k` mul, not k itself",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_tile on bounded k, got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_identity_admits_value_provable_via_prove_bounded() {
        // `np.identity(n)` where `n ∈ [0, 10]`. The bounded-admission path
        // builds a 2-D `DynamicNDArray` with envelope max_length=100
        // (n_max^2) and runtime_length aliasing the `n * n` mul.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
        let n_vid = n.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, n_vid);

        let out = np_identity(&mut b, &[n.clone()]);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                100,
                "expected prove-derived max_length=10*10 from `n <= 10`",
            );
            assert_eq!(
                data.meta.logical_shape,
                vec![10, 10],
                "expected 2-D logical_shape with n_max on each axis",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_identity on bounded n, got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_fill_multi_dim_admits_value_provable_via_prove_bounded() {
        // `np.zeros((m, 3), dtype=int)` where `m ∈ [0, 10]`. The bounded
        // axis promotes the whole shape to a 2-D `DynamicNDArray` with
        // envelope max_length=30 and runtime_length=m*3.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let m = b.ir_read_integer(InputPath::new("m", vec![]), false);
        let m_vid = m.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, m_vid);

        let three = b.ir_constant_int(3);
        let shape = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer, ZinniaType::Integer],
            values: vec![m, three],
        
            value_id: ValueId::next(),
        });

        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("dtype".to_string(), Value::Class(ZinniaType::Integer));
        let out = np_fill(&mut b, &[shape], &kwargs, 0);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                30,
                "expected prove-derived max_length=10*3 from `m <= 10`",
            );
            assert_eq!(
                data.meta.logical_shape,
                vec![10, 3],
                "expected 2-D logical_shape [m_max, 3]",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_fill on bounded (m, 3), got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_arange_3arg_admits_value_provable_via_prove_bounded() {
        // `np.arange(0, stop, 2)` where `stop ∈ [0, 10]`. len_max =
        // ceildiv(10, 2) = 5; values are [0, 2, 4, 6, 8].
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let stop = b.ir_read_integer(InputPath::new("stop", vec![]), false);
        let stop_vid = stop.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, stop_vid);
        let zero = b.ir_constant_int(0);
        let two = b.ir_constant_int(2);

        let out = np_arange(&mut b, &[zero, stop, two]);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                5,
                "expected prove-derived max_length=ceildiv(10,2)=5",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_arange(0, stop, 2) on bounded stop, got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_repeat_admits_value_provable_via_prove_bounded() {
        // `np.repeat(arr, k)` where `arr` is a 3-element static 1-D array
        // and `k ∈ [0, 10]`. Per-cell zkRAM construction: buffer length
        // `3 * 10 = 30`, runtime_length = `3 * k`.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let arr = static_floats(3);
        let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
        let k_vid = k.value_id().unwrap();
        plant_bounded_zero_to_ten(&mut b, k_vid);

        let kwargs = std::collections::HashMap::new();
        let out = ndarray_repeat(&mut b, &arr, &[k.clone()], &kwargs);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                30,
                "expected prove-derived max_length=3*10 from arr.len()=3 and `k <= 10`",
            );
            assert_ne!(
                data.meta.runtime_length.value_id, k_vid,
                "runtime_length should be the `3 * k` mul, not k itself",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_repeat on bounded k, got {:?}",
                out
            );
        }
    }

    #[test]
    fn np_linspace_admits_value_provable_via_prove_bounded() {
        // `np.linspace(0.0, 1.0, num)` where `num ∈ [2, 10]`. We need
        // `num >= 2` for endpoint=true; plant that explicitly.
        let mut b = IRBuilder::new();
        b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

        let num = b.ir_read_integer(InputPath::new("num", vec![]), false);
        let num_vid = num.value_id().unwrap();
        // Plant `num + num <= 20` (=> num <= 10) and `num >= 2`.
        let num_plus_num = ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
        };
        b.facts.insert_for(
            num_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(num_plus_num),
                rhs: Box::new(ContractTerm::LitInt(20)),
            },
        );
        b.facts.insert_for(
            num_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
                rhs: Box::new(ContractTerm::LitInt(2)),
            },
        );

        let start = Value::Float(ScalarValue::constant(0.0));
        let stop = Value::Float(ScalarValue::constant(1.0));
        let kwargs = std::collections::HashMap::new();
        let out = np_linspace(&mut b, &[start, stop, num.clone()], &kwargs);
        if let Value::DynamicNDArray(data) = &out {
            assert_eq!(
                data.max_length(),
                10,
                "expected prove-derived max_length=10 from `num <= 10`",
            );
            assert_eq!(
                data.dtype,
                crate::types::NumberType::Float,
                "default dtype should be Float",
            );
            assert_eq!(
                data.meta.runtime_length.value_id, num_vid,
                "runtime_length should alias num's value_id",
            );
        } else {
            panic!(
                "expected Value::DynamicNDArray from np_linspace on bounded num, got {:?}",
                out
            );
        }
    }
}
