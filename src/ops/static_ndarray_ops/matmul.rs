//! Matrix multiplication: `matmul`, the strategy-set dispatch
//! around it, and the shared dot-product helpers used by both
//! `matmul` itself and by external 1-D dot-product callers.

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{CompositeData, Value, ValueId};

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
