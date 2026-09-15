//! Unit tests for the statement rules of the walker: the merge of
//! `definition.md` §6.1, the loops of §6.2, `S-Unpack` and `S-SetItem` of
//! §5.11, and `S-Call` of §5.10.
//!
//! Each test builds the AST JSON the front end would export — the same serde
//! types `analyze` reads — runs the pass, and checks one property of the
//! result. The node builders live in [`super::fixtures`].

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::phase0::{Phase0Config, Phase0Result, Reject};

use super::fixtures::*;

// ---------------------------------------------------------------------------
// §6.1, the merge
// ---------------------------------------------------------------------------

#[test]
fn a_dynamic_branch_merges_the_two_lengths() {
    // if k > 0: x = a[0:2] else: x = a[0:3]; return x
    let slice = |hi: i64| {
        json!({"__class__": "ASTSubscriptExp", "val": load("a"),
               "slicing": {"__class__": "ASTSlice",
                           "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
    };
    let result = run(
        vec![array_input("a", &[8]), scalar_input("k")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("x"), slice(2))],
                vec![assign(name_target("x"), slice(3))],
            ),
            ret(load("x")),
        ],
    )
    .expect("both arms give a rank-1 integer array");
    let array = delivered(&result).as_array().expect("an array");
    assert_eq!(array.rank(), 1);
    assert_eq!(
        result.terms.constant_value(array.dims[0]),
        None,
        "the merged extent is ite(b, 2, 3), not a constant"
    );
    assert!(
        result.terms.pretty(array.dims[0]).contains("ite"),
        "the merge is an ite term, got {}",
        result.terms.pretty(array.dims[0])
    );
}

#[test]
fn a_name_bound_in_one_arm_only_is_undefined_after_the_merge() {
    let outcome = run(
        vec![scalar_input("k")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("y"), int(1))],
                vec![],
            ),
            ret(load("y")),
        ],
    );
    match outcome {
        Err(Reject::Unsupported { construct }) => assert!(
            construct.contains("only one branch"),
            "the message names the merge, got {}",
            construct
        ),
        other => panic!("expected a static error on `y`, got {:?}", other.err()),
    }
}

#[test]
fn an_if_merges_only_the_names_whose_binding_differs() {
    // if k > 0: y = a[0:2] else: y = a[0:3]; return y
    // `a`, `k` and the thirteen dtype names are bound to the same value on
    // both sides, so §6.1's merge is the identity on them: Δ records one
    // merge, for `y`, and no entry for a class name.
    let slice = |hi: i64| {
        json!({"__class__": "ASTSubscriptExp", "val": load("a"),
               "slicing": {"__class__": "ASTSlice",
                           "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
    };
    let result = run(
        vec![array_input("a", &[8]), scalar_input("k")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("y"), slice(2))],
                vec![assign(name_target("y"), slice(3))],
            ),
            ret(load("y")),
        ],
    )
    .expect("both arms give a rank-1 integer array");
    let merges: Vec<Option<String>> = result
        .descriptors
        .iter()
        .filter(|entry| {
            entry
                .producer
                .rules()
                .iter()
                .any(|rule| rule.name().starts_with("merge-"))
        })
        .map(|entry| entry.name.clone())
        .collect();
    assert_eq!(
        merges,
        vec![Some("y".to_string())],
        "only `y` differs between the arms"
    );
    let classes: Vec<String> = result
        .descriptors
        .iter()
        .filter(|entry| matches!(entry.desc, crate::phase0::desc::Desc::Class(_)))
        .map(|entry| entry.key.encode())
        .collect();
    assert!(
        classes.is_empty(),
        "a dtype name is not a value the merge produces, got {:?}",
        classes
    );
}

#[test]
fn a_scalar_no_arm_rebinds_keeps_its_declared_range() {
    // if k > 0: y = a[0:2] else: y = a[0:3]
    // for j in range(n): s = s + 1
    // `n` is bound before the `if` and neither arm touches it, so its
    // binding survives the merge and `@requires(0 <= n <= 4)` still bounds
    // the trip count. Merging it against itself would mint a fresh variable
    // with only the dtype range, and the loop would have no declared maximum.
    let slice = |hi: i64| {
        json!({"__class__": "ASTSubscriptExp", "val": load("a"),
               "slicing": {"__class__": "ASTSlice",
                           "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
    };
    let result = run_full(
        vec![array_input("a", &[8]), scalar_input("k"), scalar_input("n")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("y"), slice(2))],
                vec![assign(name_target("y"), slice(3))],
            ),
            assign(name_target("s"), int(0)),
            for_in(
                "j",
                call(None, "range", vec![load("n")]),
                vec![assign(name_target("s"), binop("add", load("s"), int(1)))],
            ),
            ret(load("s")),
        ],
        HashMap::new(),
        Phase0Config::default(),
        vec![requires_between("n", 0, 4)],
    )
    .expect("`n` keeps the range its contract declared");
    let trip_counts: Vec<String> = result
        .obligations()
        .iter()
        .filter(|o| o.kind == crate::phase0::phi::ObligationKind::Const)
        .map(|o| match o.term {
            crate::phase0::phi::ObligationTerm::Int(t) => result.terms.pretty(t),
            crate::phase0::phi::ObligationTerm::Bool(b) => result.terms.pretty_bool(b),
        })
        .collect();
    assert_eq!(
        trip_counts,
        vec!["max(0, n)".to_string()],
        "the trip count is the CONST question, over `n` itself"
    );
}

#[test]
fn the_loop_variable_of_a_bounded_range_is_guarded() {
    // for i in range(n) with `0 <= n <= 3`: an iteration the guard may skip
    // leaves `i` at its previous value, so `i` after the loop is not the
    // cap (defect 16).
    let result = run_full(
        vec![scalar_input("n")],
        vec![
            for_in("i", call(None, "range", vec![load("n")]), vec![pass_stmt()]),
            ret(load("i")),
        ],
        HashMap::new(),
        Phase0Config::default(),
        vec![requires_between("n", 0, 3)],
    )
    .expect("a requires bound is a declared maximum");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(
        result.terms.constant_value(term),
        None,
        "`i` after the loop depends on the trip count, not on the cap"
    );
    let pretty = result.terms.pretty(term);
    for j in 1..3 {
        assert!(
            pretty.contains(&format!("({} < max(0, n))", j)),
            "iteration {} does not guard the binding in {}",
            j,
            pretty
        );
    }
}

#[test]
fn a_static_condition_walks_only_the_arm_that_runs() {
    // The dead arm holds a name that is not bound anywhere; taking it would
    // reject, so accepting the program shows the arm was skipped.
    let result = run(
        vec![],
        vec![
            cond(
                binop("gt", int(1), int(0)),
                vec![assign(name_target("x"), int(7))],
                vec![assign(name_target("x"), load("never_bound"))],
            ),
            ret(load("x")),
        ],
    )
    .expect("the true arm is the only one walked");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(result.terms.constant_value(term), Some(7));
}

#[test]
fn a_merge_of_different_ranks_rejects() {
    let outcome = run(
        vec![array_input("a", &[4, 4]), scalar_input("k")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("x"), subscript(load("a"), int(0)))],
                vec![assign(name_target("x"), load("a"))],
            ),
            ret(load("x")),
        ],
    );
    assert!(
        matches!(outcome, Err(Reject::MergeMismatch { .. })),
        "§6.1's premise is equal rank, got {:?}",
        outcome.err()
    );
}

// ---------------------------------------------------------------------------
// §6.2, the loops
// ---------------------------------------------------------------------------

#[test]
fn a_range_loop_unrolls_and_folds_its_shape_terms() {
    // s = 0; for i in range(4): s = s + i; return s   ->   0 + 1 + 2 + 3 = 6
    let result = run(
        vec![],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "i",
                call(None, "range", vec![int(4)]),
                vec![assign(name_target("s"), binop("add", load("s"), load("i")))],
            ),
            ret(load("s")),
        ],
    )
    .expect("a constant range unrolls");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(result.terms.constant_value(term), Some(6));
}

#[test]
fn a_range_bound_with_no_declared_maximum_rejects() {
    // `for i in range(n)` with no `@requires` on `n`: all Φ has is the int64
    // dtype range, which §5.9 calls useless, so there is no cap to unroll to.
    let outcome = run(
        vec![scalar_input("n")],
        vec![for_in("i", call(None, "range", vec![load("n")]), vec![])],
    );
    let Err(Reject::NonConstantRangeBound { detail }) = outcome else {
        panic!(
            "the baseline makes this zero iterations (`coverage.md` §9); \
             phase 0 rejects, got {:?}",
            outcome.err()
        );
    };
    assert!(
        detail.contains("no declared maximum"),
        "the detail names what is missing, got {:?}",
        detail
    );
}

#[test]
fn a_range_bounded_by_a_requires_fact_unrolls_to_the_cap() {
    // S-For-Range-Bounded: `for i in range(2, n)` with `0 <= n <= 6` unrolls
    // 4 times, and iteration `j` runs under `j < T` with `T = max(0, n - 2)`.
    // `s = s + 1` therefore ends at `ite(0 < T, … ite(3 < T, s + 1, s) …)`,
    // whose value is not constant but whose iteration count is.
    let result = run_full(
        vec![scalar_input("n")],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "i",
                call(None, "range", vec![int(2), load("n")]),
                vec![assign(name_target("s"), binop("add", load("s"), int(1)))],
            ),
            ret(load("s")),
        ],
        HashMap::new(),
        Phase0Config::default(),
        vec![requires_between("n", 0, 6)],
    )
    .expect("a requires bound is a declared maximum");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(
        result.terms.constant_value(term),
        None,
        "the count depends on n, so the sum is a guarded term"
    );
    // The cap is `max(0, n - 2)` at `n = 6`, so iterations 0 to 3 are
    // unrolled and each is guarded by `j < T`.
    let pretty = result.terms.pretty(term);
    for j in 0..4 {
        assert!(
            pretty.contains(&format!("({} < max(0, (n - 2)))", j)),
            "iteration {} is not guarded in {}",
            j,
            pretty
        );
    }
    assert!(
        !pretty.contains("(4 < max(0, (n - 2)))"),
        "the loop unrolled past its cap: {}",
        pretty
    );
}

#[test]
fn a_bounded_range_loop_binds_its_variable_to_a_shape_term() {
    // The loop variable of iteration `j` is `t_lo + j * step`, a shape
    // scalar, so `x[i]` in the body is an index phase 0 knows (§6.2).
    let result = run_full(
        vec![array_input("x", &[8]), scalar_input("n")],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "i",
                call(None, "range", vec![int(0), load("n"), int(2)]),
                vec![assign(name_target("s"), load("i"))],
            ),
            ret(load("s")),
        ],
        HashMap::new(),
        Phase0Config::default(),
        vec![requires_between("n", 0, 5)],
    )
    .expect("a requires bound is a declared maximum");
    // ceil(5 / 2) = 3 iterations, values 0, 2, 4.
    let pretty = result.terms.pretty(
        delivered(&result)
            .as_scalar()
            .expect("a scalar")
            .term
            .as_int()
            .expect("an integer term"),
    );
    for value in ["0", "2", "4"] {
        assert!(
            pretty.contains(value),
            "iteration value {} is missing from {}",
            value,
            pretty
        );
    }
}

/// `for i in range(n)` with `0 <= n <= bound`, under a work budget.
fn bounded_loop_under_budget(bound: i64, work_budget: u64) -> Result<Phase0Result, Reject> {
    run_full(
        vec![scalar_input("n")],
        vec![for_in(
            "i",
            call(None, "range", vec![load("n")]),
            vec![assign(name_target("s"), load("i"))],
        )],
        HashMap::new(),
        Phase0Config {
            work_budget,
            ..Phase0Config::default()
        },
        vec![requires_between("n", 0, bound)],
    )
}

#[test]
fn a_cap_larger_than_the_budget_is_not_a_usable_maximum() {
    // The cap is what phase 0 would materialise, so a declared maximum past
    // the budget is reported before any of it is unrolled.
    let outcome = bounded_loop_under_budget(1024, 64);
    let Err(Reject::NonConstantRangeBound { detail }) = outcome else {
        panic!("got {:?}", outcome.err());
    };
    assert!(
        detail.contains("1024") && detail.contains("64"),
        "the detail names the cap and the budget, got {:?}",
        detail
    );
}

#[test]
fn a_bounded_range_loop_stops_at_the_work_budget() {
    // A cap the budget can afford, but a body it cannot: the rejection
    // reports the units spent and the budget, so the caller can raise it.
    let outcome = bounded_loop_under_budget(50, 64);
    let Err(Reject::BudgetExceeded { units, budget }) = outcome else {
        panic!("got {:?}", outcome.err());
    };
    assert_eq!(budget, 64);
    assert!(units > 64, "the report names the work done, got {}", units);
}

#[test]
fn a_value_sized_length_takes_its_maximum_from_a_requires_fact() {
    // E-ValueSized (§5.9): `np.zeros(n)` with `0 <= n <= 16` gets a fresh
    // `ℓ` with `0 <= ℓ <= 16`, not a rejection.
    let result = run_full(
        vec![scalar_input("n")],
        vec![ret(call(Some("np"), "zeros", vec![load("n")]))],
        HashMap::new(),
        Phase0Config::default(),
        vec![requires_between("n", 0, 16)],
    )
    .expect("a requires bound is a declared maximum");
    let array = delivered(&result).as_array().expect("an array");
    let length = array.dims[0];
    let var = result
        .terms
        .var_of(length)
        .expect("the length is the declared variable");
    let info = result.phi.var_info(var);
    let hi = info.hi.expect("ℓ has an upper bound");
    assert_eq!(result.terms.constant_value(hi), Some(16));
}

#[test]
fn a_value_sized_length_without_a_declaration_still_rejects() {
    // The same `np.zeros(n)` without the fact: the int64 dtype range is not
    // a declaration (§5.9), so the program is rejected.
    let outcome = run(
        vec![scalar_input("n")],
        vec![ret(call(Some("np"), "zeros", vec![load("n")]))],
    );
    assert!(
        matches!(outcome, Err(Reject::ValueSizedWithoutMaximum { .. })),
        "got {:?}",
        outcome.err()
    );
}

#[test]
fn a_conditional_break_leaves_later_assignments_conditional() {
    // for i in range(3): if k > 0: break; s = i
    // `s` is bound under `¬break`, so its term is an ite, not a constant.
    let result = run(
        vec![scalar_input("k")],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "i",
                call(None, "range", vec![int(3)]),
                vec![
                    cond(
                        binop("gt", load("k"), int(0)),
                        vec![json!({"__class__": "ASTBreakStatement"})],
                        vec![],
                    ),
                    assign(name_target("s"), load("i")),
                ],
            ),
            ret(load("s")),
        ],
    )
    .expect("break is a path condition, not a rejection");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(
        result.terms.constant_value(term),
        None,
        "after a conditional break the value is ite(live, i, previous)"
    );
}

#[test]
fn a_while_loop_unrolled_to_the_cap_records_the_cap_check() {
    // while k > 0: k = k - 1   with loop_limit 3: Φ += ((φ ∧ b_1 ∧ b_2 ∧ b_3) ⇒ false)
    let config = Phase0Config {
        loop_limit: 3,
        ..Phase0Config::default()
    };
    let result = run_with(
        vec![scalar_input("k")],
        vec![while_loop(
            binop("gt", load("k"), int(0)),
            vec![assign(name_target("k"), binop("sub", load("k"), int(1)))],
        )],
        HashMap::new(),
        config,
    )
    .expect("a while loop is never rejected, only capped");
    assert!(
        check_provenances(&result)
            .iter()
            .any(|p| p.contains("exit within the cap")),
        "the cap must appear as a check, got {:?}",
        check_provenances(&result)
    );
}

#[test]
fn a_while_loop_whose_guard_is_statically_false_adds_no_cap_check() {
    let result = run(
        vec![],
        vec![while_loop(binop("gt", int(0), int(1)), vec![])],
    )
    .expect("the guard folds to false");
    assert!(
        !check_provenances(&result).iter().any(|p| p.contains("cap")),
        "a loop that exits before the cap is unconstrained"
    );
}

// ---------------------------------------------------------------------------
// §5.11, unpacking and in-place writes
// ---------------------------------------------------------------------------

#[test]
fn unpacking_a_tuple_binds_element_by_element() {
    let result = run(
        vec![],
        vec![
            json!({"__class__": "ASTAssignStatement",
                   "targets": [{"__class__": "ASTTupleAssignTarget",
                                "targets": [name_target("a"), name_target("b")],
                                "star": false}],
                   "value": tuple(vec![int(3), int(5)])}),
            ret(load("b")),
        ],
    )
    .expect("two targets for two elements");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(result.terms.constant_value(term), Some(5));
}

#[test]
fn unpacking_with_the_wrong_arity_rejects() {
    let outcome = run(
        vec![],
        vec![json!({"__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTTupleAssignTarget",
                                 "targets": [name_target("a"), name_target("b")],
                                 "star": false}],
                    "value": tuple(vec![int(1), int(2), int(3)])})],
    );
    assert!(
        matches!(
            outcome,
            Err(Reject::UnpackArity {
                expected: 2,
                found: 3
            })
        ),
        "got {:?}",
        outcome.err()
    );
}

#[test]
fn a_starred_target_collects_the_middle() {
    let result = run(
        vec![],
        vec![
            json!({"__class__": "ASTAssignStatement",
                   "targets": [{"__class__": "ASTTupleAssignTarget",
                                "targets": [name_target("a"), starred_target("b"),
                                            name_target("c")],
                                "star": false}],
                   "value": tuple(vec![int(1), int(2), int(3), int(4)])}),
            ret(load("b")),
        ],
    )
    .expect("a starred target absorbs the middle two");
    let composite = delivered(&result).as_composite().expect("a composite");
    assert_eq!(composite.elems.len(), 2);
}

#[test]
fn setitem_keeps_the_descriptor_and_records_the_index_check() {
    // a[k] = 0; return a
    let result = run(
        vec![array_input("a", &[4]), scalar_input("k")],
        vec![
            json!({"__class__": "ASTAssignStatement",
                   "targets": [subscript_target("a", load("k"))],
                   "value": int(0)}),
            ret(load("a")),
        ],
    )
    .expect("an in-place write leaves the shape alone");
    assert_eq!(
        delivered_length(&result),
        Some(4),
        "S-SetItem does not change dims"
    );
    assert!(
        check_provenances(&result).iter().any(|p| p == "E-Elem"),
        "the index bound of the write is a check, got {:?}",
        check_provenances(&result)
    );
}

#[test]
fn a_composite_written_at_a_runtime_index_merges_every_element() {
    // v = [1, 2, 3]; v[k] = 9; return v
    // A runtime index may hit any element, so each becomes a merge, and the
    // conditions those merges record belong to the write (§5a).
    let result = run(
        vec![scalar_input("k")],
        vec![
            assign(
                name_target("v"),
                json!({"__class__": "ASTSquareBrackets",
                       "values": [int(1), int(2), int(3)]}),
            ),
            json!({"__class__": "ASTAssignStatement",
                   "targets": [subscript_target("v", load("k"))],
                   "value": int(9)}),
            ret(load("v")),
        ],
    )
    .expect("a composite keeps its length under a write");
    let composite = delivered(&result)
        .as_composite()
        .expect("a list was returned");
    assert_eq!(composite.elems.len(), 3, "a write changes no length");
    for (at, element) in composite.elems.iter().enumerate() {
        let term = element
            .as_scalar()
            .expect("an integer element")
            .term
            .as_int()
            .expect("an integer term");
        assert_eq!(
            result.terms.constant_value(term),
            None,
            "element {} is still its old constant",
            at
        );
    }
}

// ---------------------------------------------------------------------------
// §5.10, chip inlining
// ---------------------------------------------------------------------------

/// A chip `double(x, k=2)` returning `x * k`.
fn double_chip() -> HashMap<String, Value> {
    let chip = json!({
        "__class__": "ASTChip",
        "inputs": [{"name": "x"}, {"name": "k", "default": int(2)}],
        "block": [ret(binop("mul", load("x"), load("k")))],
        "return_dt": null
    });
    HashMap::from([(
        "double".to_string(),
        json!({"chip_ast": chip, "return_dt": null}),
    )])
}

#[test]
fn a_chip_call_inlines_the_body_and_applies_a_default() {
    let result = run_with(
        vec![],
        vec![ret(call(None, "double", vec![int(5)]))],
        double_chip(),
        Phase0Config::default(),
    )
    .expect("the default binds `k` to 2");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    let term = scalar.term.as_int().expect("an integer term");
    assert_eq!(result.terms.constant_value(term), Some(10));
}

#[test]
fn recursion_deeper_than_the_limit_rejects() {
    // A chip that calls itself unconditionally: the depth bound is the only
    // thing that stops the inlining (§5.10).
    let chip = json!({
        "__class__": "ASTChip",
        "inputs": [{"name": "x"}],
        "block": [ret(call(None, "loop", vec![load("x")]))],
        "return_dt": null
    });
    let chips = HashMap::from([(
        "loop".to_string(),
        json!({"chip_ast": chip, "return_dt": null}),
    )]);
    let config = Phase0Config {
        recursion_limit: 4,
        ..Phase0Config::default()
    };
    let outcome = run_with(
        vec![],
        vec![ret(call(None, "loop", vec![int(1)]))],
        chips,
        config,
    );
    assert!(
        matches!(outcome, Err(Reject::RecursionTooDeep { limit: 4 })),
        "got {:?}",
        outcome.err()
    );
}

// ---------------------------------------------------------------------------
// Corrections from the step-6 review
// ---------------------------------------------------------------------------

/// A chip whose declared return type stands in for a body that never
/// returns, so that `inline_chip` takes its `E-External` fallback.
fn declared_return_chip() -> HashMap<String, Value> {
    let chip = json!({
        "__class__": "ASTChip",
        "inputs": [],
        "block": [pass_stmt()],
        "return_dt": {"NDArray": {"shape": [3], "dtype": "Integer"}}
    });
    HashMap::from([(
        "make".to_string(),
        json!({"chip_ast": chip,
               "return_dt": {"NDArray": {"shape": [3], "dtype": "Integer"}}}),
    )])
}

#[test]
fn a_chip_called_inside_a_branch_is_guarded_by_the_path_condition() {
    // `inline_chip`'s fallback built its descriptor under `true`, so Δ
    // claimed the value existed on every path (§2: ψ is φ at creation).
    let result = run_with(
        vec![scalar_input("k")],
        vec![cond(
            binop("gt", load("k"), int(0)),
            vec![ret(call(None, "make", vec![]))],
            vec![],
        )],
        declared_return_chip(),
        Phase0Config::default(),
    )
    .expect("the chip's declared return type gives the value a descriptor");
    let array = delivered(&result).as_array().expect("an array");
    // The branch condition is a data Boolean, so ψ is that variable; all the
    // test needs is that it is not the unconditional `true` the fallback
    // used to hand out.
    assert!(
        !result.terms.is_true(array.psi),
        "ψ is `{}`, not the branch condition",
        result.terms.pretty_bool(array.psi)
    );
}

#[test]
fn each_element_of_a_composite_input_takes_its_own_value_key() {
    // Two scalar elements of one tuple input used to share the parent's key,
    // so θ(u) could not tell their variables apart (§5.1, `E-Input-Composite`).
    let input = json!({"name": "p",
                       "annotation": {"dt": {"Tuple": {"elements": ["Integer", "Integer"]}}}});
    let result = run(vec![input], vec![]).expect("a tuple of two scalar inputs");
    let keys: Vec<String> = result
        .phi
        .vars()
        .iter()
        .filter_map(|v| match &v.origin {
            crate::phase0::phi::VarOrigin::DataScalar { value } => Some(value.encode()),
            _ => None,
        })
        .collect();
    assert_eq!(keys, vec!["|0||0".to_string(), "|1||0".to_string()]);
}

#[test]
fn a_constant_range_past_the_materialisation_limit_is_a_budget_rejection() {
    // Not `Unsupported`: `range` has a rule (§6.2), and what fails is the
    // number of elements phase 0 will materialise.
    let outcome = run(
        vec![],
        vec![for_in(
            "i",
            call(None, "range", vec![int(5_000_000)]),
            vec![pass_stmt()],
        )],
    );
    match outcome {
        Err(Reject::BudgetExceeded { units, budget }) => {
            assert_eq!(units, 5_000_000);
            assert_eq!(budget, 1 << 22);
        }
        other => panic!("got {:?}", other.err()),
    }
}
