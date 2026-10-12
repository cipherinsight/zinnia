// ────────────────────────────────────────────────────────────────────────
// Tests — bound-aware reshape / split chokepoints
// (compiler.bound-aware-reshape, compiler.bound-aware-split)
// ────────────────────────────────────────────────────────────────────────

use super::*;
use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
use crate::optim::resolver::LayeredResolver;
use crate::types::{CompositeData, NumberType, ScalarValue, Value, ValueId, ZinniaType};

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

fn static_int_array(b: &mut IRBuilder, vals: &[i64], shape: Vec<usize>) -> Value {
    let flat = vals.iter().map(|&v| b.ir_constant_int(v)).collect();
    crate::helpers::static_array::base::build_static_array_from_flat(
        b,
        flat,
        shape,
        NumberType::Integer,
    )
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));
    let arr = static_floats(8); // 1-D of length 8

    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();
    plant_k_squared_eq(&mut b, k_vid, 16); // k == 4
    let two = b.ir_constant_int(2);

    // reshape(arr, k, 2) → shape (4, 2).
    let out = ndarray_reshape(&mut b, &arr, &[k, two]);
    let shape = crate::helpers::composite::get_composite_shape(&out);
    assert_eq!(
        shape,
        vec![4, 2],
        "reshape did not produce the prove-derived shape"
    );
}

#[test]
fn split_admits_value_provable_via_prove() {
    // np.split(arr, k) with k == 2 (derived from `k * k == 4`).
    let mut b = IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));
    let arr = static_floats(8);
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let two = b.ir_constant_int(2);
    let _ = ndarray_reshape(&mut b, &arr, &[k, two]);
}

#[test]
#[should_panic(expected = "split sections must be a compile-time constant int")]
fn split_still_rejects_when_no_facts() {
    let mut b = IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
        panic!(
            "expected Value::DynamicNDArray from np_fill on bounded k, got {:?}",
            out
        );
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

    let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let n_vid = n.value_id().unwrap();
    plant_bounded_zero_to_ten(&mut b, n_vid);

    let out = np_arange(&mut b, &[n.clone()], &std::collections::HashMap::new());
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
fn np_sort_static_array_sorts_values_and_fires_sorted_fact() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[3, 1, 2], vec![3]);
    let out =
        crate::helpers::static_array::sorting::try_apply_sort(&mut b, &arr, None, false).unwrap();
    let vals: Vec<Option<i64>> =
        crate::helpers::static_array::elementwise::payload_cells(&mut b, &out)
            .iter()
            .map(|v| v.int_val())
            .collect();
    assert_eq!(vals, vec![Some(1), Some(2), Some(3)]);
    // The sorted array's order is an array fact of its contents.
    assert!(b.array_has_order(
        &out,
        crate::optim::array_facts::Order::NonDecreasing,
        crate::builder::Where::Everywhere
    ));
}

#[test]
fn np_sort_of_runtime_values_states_a_non_decreasing_order() {
    let mut b = IRBuilder::new();
    let cells: Vec<Value> = (0..3)
        .map(|k| b.ir_read_integer(crate::circuit_input::InputPath::new(&format!("x{k}"), vec![]), false))
        .collect();
    let arr = crate::helpers::static_array::base::build_static_array_from_flat(&mut b, cells, vec![3], NumberType::Integer);
    let out = crate::helpers::static_array::sorting::try_apply_sort(&mut b, &arr, None, false).unwrap();
    use crate::optim::array_facts::Order;
    assert!(b.array_has_order(&out, Order::NonDecreasing, crate::builder::Where::Everywhere));
    assert!(!b.array_has_order(&out, Order::Increasing, crate::builder::Where::Everywhere), "ties are possible");
    let index = crate::helpers::static_array::sorting::try_apply_sort(&mut b, &arr, None, true).unwrap();
    let lanes = crate::helpers::static_array::elementwise::payload_cells(&mut b, &index);
    assert!(b.is_known_permutation(&lanes), "argsort lanes are a permutation");
}

#[test]
fn np_argsort_static_array_is_stable_for_equal_values() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[3, 1, 2, 1], vec![4]);
    let out =
        crate::helpers::static_array::sorting::try_apply_sort(&mut b, &arr, None, true).unwrap();
    let vals: Vec<Option<i64>> =
        crate::helpers::static_array::elementwise::payload_cells(&mut b, &out)
            .iter()
            .map(|v| v.int_val())
            .collect();
    assert_eq!(vals, vec![Some(1), Some(3), Some(2), Some(0)]);
}

#[test]
fn np_take_static_array_axis_one_and_negative_indices() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[10, 20, 30, 40, 50, 60], vec![2, 3]);
    let idx = Value::List(CompositeData {
        elements_type: vec![ZinniaType::Integer; 2],
        values: vec![b.ir_constant_int(-1), b.ir_constant_int(0)],
        value_id: ValueId::next(),
    });
    let axis = b.ir_constant_int(1);
    let out =
        crate::helpers::static_array::sorting::try_apply_take(&mut b, &arr, &idx, Some(&axis))
            .unwrap();
    match &out {
        Value::StaticArray { shape, .. } => assert_eq!(shape, &vec![2, 2]),
        _ => panic!("expected StaticArray"),
    }
    let vals: Vec<Option<i64>> =
        crate::helpers::static_array::elementwise::payload_cells(&mut b, &out)
            .iter()
            .map(|v| v.int_val())
            .collect();
    assert_eq!(vals, vec![Some(30), Some(10), Some(60), Some(40)]);
}

#[test]
fn np_full_static_known_fill_is_constant() {
    // The cells are the fill constant: every element is 7 (array facts read
    // it off them).
    let mut b = IRBuilder::new();
    let shape = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; 2],
        values: vec![b.ir_constant_int(2), b.ir_constant_int(2)],
        value_id: ValueId::next(),
    });
    let fill = b.ir_constant_int(7);
    let out = np_full(&mut b, &[shape, fill], &std::collections::HashMap::new());
    assert!(b.array_all_equal(&out, 7));
    assert!(!b.array_all_equal(&out, 0));
}

#[test]
#[should_panic(expected = "np.full/full_like: Complex fill is not yet implemented")]
fn np_full_rejects_explicit_complex_dtype() {
    let mut b = IRBuilder::new();
    let shape = Value::Tuple(CompositeData {
        elements_type: vec![ZinniaType::Integer; 1],
        values: vec![b.ir_constant_int(2)],
        value_id: ValueId::next(),
    });
    let fill = b.ir_constant_int(1);
    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("dtype".to_string(), Value::Class(ZinniaType::Complex));
    let _ = np_full(&mut b, &[shape, fill], &kwargs);
}

#[test]
#[should_panic(expected = "np.full/full_like: Complex fill is not yet implemented")]
fn np_full_like_rejects_explicit_complex_dtype() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[1, 2], vec![2]);
    let fill = b.ir_constant_int(1);
    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("dtype".to_string(), Value::Class(ZinniaType::Complex));
    let _ = np_full_like(&mut b, &[arr, fill], &kwargs);
}

#[test]
fn np_var_std_static_array_axis_and_ddof_are_native() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[1, 2, 3, 4], vec![2, 2]);
    let axis = b.ir_constant_int(1);
    let ddof = b.ir_constant_int(1);
    let out = crate::helpers::static_array::reductions::try_apply_var_std(
        &mut b,
        "var",
        &arr,
        Some(&axis),
        Some(&ddof),
        false,
    )
    .expect("native var");
    match &out {
        Value::StaticArray { shape, dtype, .. } => {
            assert_eq!(shape, &vec![2]);
            assert_eq!(*dtype, NumberType::Float);
        }
        _ => panic!("expected StaticArray"),
    }
    let vals: Vec<Option<f64>> =
        crate::helpers::static_array::elementwise::payload_cells(&mut b, &out)
            .iter()
            .map(|v| v.float_val())
            .collect();
    assert_eq!(vals, vec![Some(0.5), Some(0.5)]);

    let std_out = crate::helpers::static_array::reductions::try_apply_var_std(
        &mut b, "std", &arr, None, None, false,
    )
    .expect("native std");
    assert!(matches!(
        std_out,
        Value::StaticArray {
            dtype: NumberType::Float,
            ..
        }
    ));
}

#[test]
#[should_panic(expected = "non-positive divisor")]
fn np_var_static_array_rejects_invalid_ddof_for_axis_fiber() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[1, 2, 3, 4], vec![2, 2]);
    let axis = b.ir_constant_int(1);
    let ddof = b.ir_constant_int(2);
    let _ = crate::helpers::static_array::reductions::try_apply_var_std(
        &mut b,
        "var",
        &arr,
        Some(&axis),
        Some(&ddof),
        false,
    );
}

#[test]
fn np_cov_static_array_rowvar_true_default() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[1, 2, 3, 2, 4, 6], vec![2, 3]);
    let out = np_cov(&mut b, &[arr], &std::collections::HashMap::new());
    match &out {
        Value::StaticArray { shape, dtype, .. } => {
            assert_eq!(shape, &vec![2, 2]);
            assert_eq!(*dtype, NumberType::Float);
        }
        _ => panic!("expected StaticArray"),
    }
    let vals: Vec<Option<f64>> =
        crate::helpers::static_array::elementwise::payload_cells(&mut b, &out)
            .iter()
            .map(|v| v.float_val())
            .collect();
    assert_eq!(vals, vec![Some(1.0), Some(2.0), Some(2.0), Some(4.0)]);
}

#[test]
#[should_panic(expected = "weights")]
fn np_cov_rejects_weight_kwargs() {
    let mut b = IRBuilder::new();
    let arr = static_int_array(&mut b, &[1, 2, 3, 4], vec![2, 2]);
    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("fweights".to_string(), b.ir_constant_int(1));
    let _ = np_cov(&mut b, &[arr], &kwargs);
}

#[test]
fn np_linalg_verify_inverse_emits_bidirectional_asserts() {
    let mut b = IRBuilder::new();
    let a = static_int_array(&mut b, &[1, 0, 0, 1], vec![2, 2]);
    let y = static_int_array(&mut b, &[1, 0, 0, 1], vec![2, 2]);
    let out = np_linalg_verify_inverse(&mut b, &[a, y.clone()]);
    assert_eq!(
        crate::helpers::composite::get_composite_shape(&out),
        vec![2, 2]
    );
    let assert_count = b
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, crate::ir_defs::IR::Assert))
        .count();
    assert_eq!(assert_count, 2, "A@Y and Y@A must both be asserted");
}

#[test]
fn np_linalg_verify_solve_accepts_vector_and_matrix_rhs() {
    let mut b = IRBuilder::new();
    let a = static_int_array(&mut b, &[1, 0, 0, 1], vec![2, 2]);
    let x = static_int_array(&mut b, &[3, 4], vec![2]);
    let rhs = static_int_array(&mut b, &[3, 4], vec![2]);
    let out = np_linalg_verify_solve(&mut b, &[a.clone(), x.clone(), rhs]);
    assert_eq!(
        crate::helpers::composite::get_composite_shape(&out),
        vec![2]
    );

    let x_matrix = static_int_array(&mut b, &[3, 4, 5, 6], vec![2, 2]);
    let rhs_matrix = static_int_array(&mut b, &[3, 4, 5, 6], vec![2, 2]);
    let out = np_linalg_verify_solve(&mut b, &[a, x_matrix.clone(), rhs_matrix]);
    assert_eq!(
        crate::helpers::composite::get_composite_shape(&out),
        vec![2, 2]
    );
}

#[test]
fn np_linalg_verify_cholesky_asserts_lower_triangle_and_reconstruction() {
    let mut b = IRBuilder::new();
    let a = static_int_array(&mut b, &[4, 2, 2, 5], vec![2, 2]);
    let l = static_int_array(&mut b, &[2, 0, 1, 2], vec![2, 2]);
    let out = np_linalg_verify_cholesky(&mut b, &[a, l.clone()]);
    assert_eq!(
        crate::helpers::composite::get_composite_shape(&out),
        vec![2, 2]
    );
    let assert_count = b
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, crate::ir_defs::IR::Assert))
        .count();
    assert_eq!(
        assert_count, 2,
        "one upper-triangle zero plus one reconstruction assert expected for 2x2"
    );
}

#[test]
#[should_panic(expected = "expected x and b to both be shape")]
fn np_linalg_verify_solve_rejects_mismatched_rhs_shapes() {
    let mut b = IRBuilder::new();
    let a = static_int_array(&mut b, &[1, 0, 0, 1], vec![2, 2]);
    let x = static_int_array(&mut b, &[3, 4], vec![2]);
    let rhs = static_int_array(&mut b, &[3, 4, 5], vec![3]);
    let _ = np_linalg_verify_solve(&mut b, &[a, x, rhs]);
}

#[test]
#[should_panic(expected = "implicit result-returning linear algebra is deferred")]
fn np_linalg_inv_implicit_result_rejects_with_helper_hint() {
    let _ = np_linalg_implicit_result("inv");
}

#[test]
fn np_arange_3arg_admits_value_provable_via_prove_bounded() {
    // `np.arange(0, stop, 2)` where `stop ∈ [0, 10]`. len_max =
    // ceildiv(10, 2) = 5; values are [0, 2, 4, 6, 8].
    let mut b = IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

    let stop = b.ir_read_integer(InputPath::new("stop", vec![]), false);
    let stop_vid = stop.value_id().unwrap();
    plant_bounded_zero_to_ten(&mut b, stop_vid);
    let zero = b.ir_constant_int(0);
    let two = b.ir_constant_int(2);

    let out = np_arange(&mut b, &[zero, stop, two], &std::collections::HashMap::new());
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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(
        crate::optim::z3_budget::OPTIMIZER_RESOLVER_RLIMIT, 4096,
    )));

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
