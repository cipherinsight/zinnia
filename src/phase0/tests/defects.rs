//! The reproductions of the defect list in `revision/phase0/implementation.md`,
//! one test per defect.
//!
//! Each test is the program that row names, written as the AST JSON the front
//! end exports, with the shape NumPy gives it as the expectation. They are
//! regression tests: every one of them failed against the step-2b build.

use serde_json::{json, Value};

use crate::phase0::desc::{Desc, Dtype};
use crate::phase0::phi::{ConstraintKind, ObligationKind};
use crate::phase0::{Phase0Result, Reject};

use super::fixtures::*;

// ---------------------------------------------------------------------------
// Helpers specific to these programs
// ---------------------------------------------------------------------------

/// A fixed-shape float array input, for the rules whose result is real.
fn float_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Float"}}}})
}

fn call_kw(target: Option<&str>, member: &str, args: Vec<Value>, kwargs: Value) -> Value {
    json!({"__class__": "ASTNamedAttribute", "target": target, "member": member,
           "args": args, "kwargs": kwargs})
}

fn method(receiver: Value, member: &str, args: Vec<Value>) -> Value {
    json!({"__class__": "ASTExprAttribute", "target": receiver, "member": member,
           "args": args, "kwargs": {}})
}

fn list(values: Vec<Value>) -> Value {
    json!({"__class__": "ASTSquareBrackets", "values": values})
}

/// `a[i_0, ..., i_k]`, a multi-axis subscript.
fn subscript_many(value: Value, entries: Vec<Value>) -> Value {
    json!({"__class__": "ASTSubscriptExp", "val": value,
           "slicing": {"__class__": "ASTSlice", "data": entries}})
}

/// The constant extents of the delivered array, or `None` where one is a term.
fn delivered_dims(result: &Phase0Result) -> Vec<Option<i128>> {
    let array = delivered(result).as_array().expect("an array was returned");
    array
        .dims
        .iter()
        .map(|&d| result.terms.constant_value(d))
        .collect()
}

/// Every check `Φ` carries, pretty-printed with its provenance.
fn checks(result: &Phase0Result) -> Vec<String> {
    result
        .phi
        .constraints()
        .iter()
        .filter(|c| matches!(c.kind, ConstraintKind::Check { .. }))
        .map(|c| format!("{}: {}", c.provenance, result.terms.pretty_bool(c.formula)))
        .collect()
}

fn obligations_of(result: &Phase0Result, kind: ObligationKind) -> Vec<String> {
    result
        .obligations()
        .iter()
        .filter(|o| o.kind == kind)
        .map(|o| o.provenance.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// 1. `E-Take` broadcasts several advanced indices (§5.4)
// ---------------------------------------------------------------------------

#[test]
fn several_advanced_indices_broadcast_into_one_block() {
    // b[i, j] with b 3x3 and i, j of length 3: NumPy gives (3,).
    let result = run(
        vec![
            array_input("b", &[3, 3]),
            array_input("i", &[3]),
            array_input("j", &[3]),
        ],
        vec![ret(subscript_many(load("b"), vec![load("i"), load("j")]))],
    )
    .expect("a two-index gather is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(3)]);
}

#[test]
fn an_advanced_index_separated_by_a_slice_moves_the_block_to_the_front() {
    // b[i, :, j] with b 4x5x6 and i, j of length 3: NumPy gives (3, 5).
    let full_slice = json!([
        {"__class__": "ASTConstantNone"},
        {"__class__": "ASTConstantNone"},
        {"__class__": "ASTConstantNone"}
    ]);
    let result = run(
        vec![
            array_input("b", &[4, 5, 6]),
            array_input("i", &[3]),
            array_input("j", &[3]),
        ],
        vec![ret(subscript_many(
            load("b"),
            vec![load("i"), full_slice, load("j")],
        ))],
    )
    .expect("a separated gather is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(3), Some(5)]);
}

#[test]
fn one_advanced_index_still_keeps_its_place() {
    // b[:, i] with b 4x5 and i of length 3: NumPy gives (4, 3), which is
    // `E-Take` on axis 1 exactly as §5.4 states it.
    let full_slice = json!([
        {"__class__": "ASTConstantNone"},
        {"__class__": "ASTConstantNone"},
        {"__class__": "ASTConstantNone"}
    ]);
    let result = run(
        vec![array_input("b", &[4, 5]), array_input("i", &[3])],
        vec![ret(subscript_many(load("b"), vec![full_slice, load("i")]))],
    )
    .expect("a single gather is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(4), Some(3)]);
}

// ---------------------------------------------------------------------------
// 2. `E-Reshape`'s declared axis needs a positive product (§5.6)
// ---------------------------------------------------------------------------

#[test]
fn a_declared_reshape_axis_requires_a_positive_product() {
    // b = a[:k]; b.reshape(-1, k): with k = 0 NumPy raises, so `defcond` has
    // to be false there, which `size(a) = l * k` alone is not.
    let slice_to_k = json!({"__class__": "ASTSubscriptExp", "val": load("a"),
        "slicing": {"__class__": "ASTSlice", "data": [[
            {"__class__": "ASTConstantNone"}, load("k"),
            {"__class__": "ASTConstantNone"}]]}});
    let result = run(
        vec![float_input("a", &[4]), scalar_input("k")],
        vec![
            assign(name_target("b"), slice_to_k),
            ret(method(load("b"), "reshape", vec![int(-1), load("k")])),
        ],
    )
    .expect("a declared reshape axis is in scope");
    let reshape: Vec<String> = checks(&result)
        .into_iter()
        .filter(|text| text.starts_with("E-Reshape"))
        .collect();
    assert_eq!(reshape.len(), 1, "one reshape check, got {:?}", reshape);
    assert!(
        reshape[0].contains("0 < "),
        "the check demands a positive product, got {}",
        reshape[0]
    );
}

// ---------------------------------------------------------------------------
// 3. `np.insert` bounds its position (`coverage.md` §5.3)
// ---------------------------------------------------------------------------

#[test]
fn np_insert_bounds_its_position() {
    let result = run(
        vec![float_input("a", &[4]), scalar_input("pos")],
        vec![ret(call(
            Some("np"),
            "insert",
            vec![load("a"), load("pos"), int(1)],
        ))],
    )
    .expect("np.insert is in scope");
    let insert: Vec<String> = checks(&result)
        .into_iter()
        .filter(|text| text.starts_with("E-Insert"))
        .collect();
    assert_eq!(insert.len(), 1, "one insert check, got {:?}", insert);
    assert!(
        insert[0].contains("-4 <= pos") && insert[0].contains("pos <= 4"),
        "the position is bounded by the axis, got {}",
        insert[0]
    );
    assert_eq!(delivered_dims(&result), vec![Some(5)]);
}

// ---------------------------------------------------------------------------
// 4. `np.ravel_multi_index` records a definedness condition (§5.4)
// ---------------------------------------------------------------------------

#[test]
fn ravel_multi_index_records_a_definedness_condition() {
    let result = run(
        vec![array_input("index", &[2]), array_input("dims", &[2])],
        vec![ret(call_kw(
            Some("np"),
            "ravel_multi_index",
            vec![load("index")],
            json!({"dims": load("dims")}),
        ))],
    )
    .expect("np.ravel_multi_index is in scope");
    assert!(
        checks(&result)
            .iter()
            .any(|text| text.contains("in_ravel_multi_index")),
        "a coordinate condition is recorded, got {:?}",
        checks(&result)
    );
}

// ---------------------------------------------------------------------------
// 5. `np.percentile` records its domain (`coverage.md` §5.2)
// ---------------------------------------------------------------------------

#[test]
fn percentile_records_its_domain() {
    let result = run(
        vec![float_input("a", &[4]), scalar_input("p")],
        vec![ret(call(
            Some("np"),
            "percentile",
            vec![load("a"), load("p")],
        ))],
    )
    .expect("np.percentile is in scope");
    let domain: Vec<String> = checks(&result)
        .into_iter()
        .filter(|text| text.contains("100"))
        .collect();
    assert_eq!(domain.len(), 1, "one domain check, got {:?}", domain);
    assert!(
        domain[0].contains("0 <= p") && domain[0].contains("p <= 100"),
        "the domain is [0, 100], got {}",
        domain[0]
    );
}

// ---------------------------------------------------------------------------
// 6. `E-Map` records the domain condition of its operator (§5.3)
// ---------------------------------------------------------------------------

#[test]
fn math_sqrt_of_a_shape_scalar_records_its_domain() {
    // `math.sqrt(len(a) - 8)` is negative for every admitted `a`, and the
    // check says so as a term rather than as an opaque Boolean.
    let result = run(
        vec![array_input("a", &[4])],
        vec![ret(call(
            Some("math"),
            "sqrt",
            vec![binop("sub", call(None, "len", vec![load("a")]), int(8))],
        ))],
    )
    .expect("math.sqrt is in scope");
    let domain: Vec<String> = checks(&result)
        .into_iter()
        .filter(|text| text.starts_with("E-Map"))
        .collect();
    assert_eq!(domain.len(), 1, "one domain check, got {:?}", domain);
    assert!(
        domain[0].contains("false"),
        "`sqrt(-4)` is out of the domain, got {}",
        domain[0]
    );
}

#[test]
fn a_map_over_an_array_records_an_unconstrained_domain_witness() {
    // Φ does not model element values (§1), so the domain condition of an
    // elementwise `log` is a fresh Boolean, as `E-Take`'s `inrange` is.
    let result = run(
        vec![float_input("a", &[4])],
        vec![ret(call(Some("np"), "log", vec![load("a")]))],
    )
    .expect("np.log is in scope");
    assert!(
        checks(&result)
            .iter()
            .any(|text| text.starts_with("E-Map") && text.contains("in_domain")),
        "a domain witness is recorded, got {:?}",
        checks(&result)
    );
}

#[test]
fn an_operator_with_no_domain_records_no_check() {
    let result = run(
        vec![float_input("a", &[4])],
        vec![ret(call(Some("np"), "exp", vec![load("a")]))],
    )
    .expect("np.exp is in scope");
    assert!(
        checks(&result).is_empty(),
        "`exp` is total, got {:?}",
        checks(&result)
    );
}

// ---------------------------------------------------------------------------
// 7. `np.correlate`'s length (`coverage.md` §5.6)
// ---------------------------------------------------------------------------

#[test]
fn correlate_is_symmetric_in_its_operand_lengths() {
    for (n, m) in [(4usize, 9usize), (9, 4)] {
        let result = run(
            vec![float_input("a", &[n]), float_input("v", &[m])],
            vec![ret(call(
                Some("np"),
                "correlate",
                vec![load("a"), load("v")],
            ))],
        )
        .expect("np.correlate is in scope");
        assert_eq!(
            delivered_dims(&result),
            vec![Some(6)],
            "|{} - {}| + 1 = 6",
            n,
            m
        );
    }
}

// ---------------------------------------------------------------------------
// 8. The two arms of an `if` get distinct keys (`acceptance.md` §2.2)
// ---------------------------------------------------------------------------

#[test]
fn the_two_arms_of_an_if_get_distinct_keys() {
    // Both arms bind `a` at the same statement index; Δ must keep both.
    let slice = |hi: i64| {
        json!({"__class__": "ASTSubscriptExp", "val": load("x"),
               "slicing": {"__class__": "ASTSlice",
                           "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
    };
    let result = run(
        vec![array_input("x", &[8]), scalar_input("f")],
        vec![
            cond(
                binop("gt", load("f"), int(0)),
                vec![assign(name_target("a"), slice(2))],
                vec![assign(name_target("a"), slice(5))],
            ),
            ret(method(load("a"), "sum", vec![])),
        ],
    )
    .expect("both arms bind a rank-1 array");
    let bound: Vec<Vec<Option<i128>>> = result
        .descriptors
        .iter()
        .filter(|entry| entry.name.as_deref() == Some("a"))
        .filter_map(|entry| entry.desc.as_array())
        .map(|array| {
            array
                .dims
                .iter()
                .map(|&d| result.terms.constant_value(d))
                .collect()
        })
        .collect();
    assert!(
        bound.contains(&vec![Some(2)]) && bound.contains(&vec![Some(5)]),
        "Δ keeps both arms' `a`, got {:?}",
        bound
    );
    let keys: Vec<String> = result
        .descriptors
        .iter()
        .filter(|entry| entry.name.as_deref() == Some("a"))
        .map(|entry| entry.key.encode())
        .collect();
    assert!(
        keys.iter().any(|k| k.starts_with("0.then.")),
        "the `then` arm is named in the key, got {:?}",
        keys
    );
    assert!(
        keys.iter().any(|k| k.starts_with("0.else.")),
        "the `else` arm is named in the key, got {:?}",
        keys
    );
}

#[test]
fn the_two_arms_of_a_conditional_expression_get_distinct_keys() {
    // `y = x[:2] if f > 0 else x[:5]`: the arms are expressions, and the
    // merged descriptor must see two different lengths.
    let slice = |hi: i64| {
        json!({"__class__": "ASTSubscriptExp", "val": load("x"),
               "slicing": {"__class__": "ASTSlice",
                           "data": [[int(0), int(hi), {"__class__": "ASTConstantNone"}]]}})
    };
    let cond_exp = json!({"__class__": "ASTCondExp",
        "cond": binop("gt", load("f"), int(0)),
        "t_expr": slice(2), "f_expr": slice(5)});
    let result = run(
        vec![array_input("x", &[8]), scalar_input("f")],
        vec![assign(name_target("y"), cond_exp), ret(load("y"))],
    )
    .expect("both arms give a rank-1 array");
    let array = delivered(&result).as_array().expect("an array");
    assert!(
        result.terms.pretty(array.dims[0]).contains("ite"),
        "the two arms merge, got {}",
        result.terms.pretty(array.dims[0])
    );
}

// ---------------------------------------------------------------------------
// 9. A bounded iteration guards what it creates (§6.2, §2)
// ---------------------------------------------------------------------------

#[test]
fn a_bounded_range_loop_guards_its_loop_variable() {
    // `for i in range(n)` with `0 <= n <= 4`: iteration 3 runs under
    // `3 < max(0, n)`, so the `i` it binds must carry that in ψ.
    let result = run_full(
        vec![scalar_input("n")],
        vec![
            for_in(
                "i",
                call(None, "range", vec![load("n")]),
                vec![assign(name_target("s"), load("i"))],
            ),
            ret(int(0)),
        ],
        Default::default(),
        crate::phase0::Phase0Config::default(),
        vec![requires_between("n", 0, 4)],
    )
    .expect("a bounded range unrolls");
    let last = result
        .descriptors
        .iter()
        .filter(|entry| entry.name.as_deref() == Some("i"))
        .filter(|entry| entry.key.iters == vec![3])
        .map(|entry| {
            result
                .terms
                .pretty_bool(entry.desc.psi().expect("a scalar"))
        })
        .next()
        .expect("iteration 3 binds `i`");
    assert!(
        last.contains("3 <"),
        "ψ carries the iteration's guard, got {}",
        last
    );
}

#[test]
fn a_constant_range_loop_leaves_psi_unconditional() {
    let result = run(
        vec![],
        vec![
            for_in(
                "i",
                call(None, "range", vec![int(3)]),
                vec![assign(name_target("s"), load("i"))],
            ),
            ret(int(0)),
        ],
    )
    .expect("a constant range unrolls");
    for entry in &result.descriptors {
        if entry.name.as_deref() == Some("i") {
            let psi = entry.desc.psi().expect("a scalar");
            assert!(
                result.terms.is_true(psi),
                "a constant trip count needs no guard, got {}",
                result.terms.pretty_bool(psi)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// A composite's length is a compile-time constant (§3.3)
// ---------------------------------------------------------------------------

#[test]
fn a_list_method_under_a_runtime_condition_rejects() {
    let outcome = run_full(
        vec![scalar_input("n")],
        vec![
            assign(name_target("v"), list(vec![])),
            for_in(
                "i",
                call(None, "range", vec![load("n")]),
                vec![json!({"__class__": "ASTExprStatement",
                            "expr": call(Some("v"), "append", vec![load("i")])})],
            ),
            ret(call(None, "len", vec![load("v")])),
        ],
        Default::default(),
        crate::phase0::Phase0Config::default(),
        vec![requires_between("n", 0, 4)],
    );
    match outcome {
        Err(Reject::ValueSizedWithoutMaximum { construct }) => {
            assert!(
                construct.contains("`v`") && construct.contains("§3.3"),
                "the message names the list and the rule, got {}",
                construct
            );
        }
        other => panic!("expected a rejection, got {:?}", other.err()),
    }
}

#[test]
fn a_list_method_under_a_constant_condition_is_accepted() {
    // The guard of `for i in range(3)` folds to true, so the list's length
    // stays a compile-time constant and nothing is rejected.
    let result = run(
        vec![],
        vec![
            assign(name_target("v"), list(vec![])),
            for_in(
                "i",
                call(None, "range", vec![int(3)]),
                vec![json!({"__class__": "ASTExprStatement",
                            "expr": call(Some("v"), "append", vec![load("i")])})],
            ),
            ret(call(None, "len", vec![load("v")])),
        ],
    )
    .expect("a constant trip count keeps the length static");
    let length = delivered(&result).as_scalar().expect("a scalar");
    let term = length.term.as_int().expect("an integer term");
    assert_eq!(result.terms.constant_value(term), Some(3));
}

// ---------------------------------------------------------------------------
// 10. `tolist` is a nested composite (`coverage.md` §5.3)
// ---------------------------------------------------------------------------

#[test]
fn tolist_is_a_nested_composite() {
    let result = run(
        vec![array_input("a", &[2, 2])],
        vec![ret(method(load("a"), "tolist", vec![]))],
    )
    .expect("tolist is in scope");
    let outer = delivered(&result).as_composite().expect("a list");
    assert_eq!(outer.elems.len(), 2);
    for row in &outer.elems {
        let inner = row.as_composite().expect("a nested list");
        assert_eq!(inner.elems.len(), 2);
        assert!(matches!(inner.elems[0], Desc::Scalar(_)));
    }
}

#[test]
fn tolist_of_a_runtime_extent_rejects() {
    let slice_to_k = json!({"__class__": "ASTSubscriptExp", "val": load("a"),
        "slicing": {"__class__": "ASTSlice", "data": [[
            {"__class__": "ASTConstantNone"}, load("k"),
            {"__class__": "ASTConstantNone"}]]}});
    let outcome = run(
        vec![array_input("a", &[4]), scalar_input("k")],
        vec![ret(method(slice_to_k, "tolist", vec![]))],
    );
    match outcome {
        Err(Reject::ValueSizedWithoutMaximum { construct }) => assert!(
            construct.contains("§3.3"),
            "the message cites the rule, got {}",
            construct
        ),
        other => panic!("expected a rejection, got {:?}", other.err()),
    }
}

// ---------------------------------------------------------------------------
// 11. `np.dstack` promotes with `atleast_3d` (`coverage.md` §5.3)
// ---------------------------------------------------------------------------

#[test]
fn dstack_promotes_a_rank_one_operand_to_one_n_one() {
    let result = run(
        vec![array_input("a", &[3]), array_input("b", &[3])],
        vec![ret(call(
            Some("np"),
            "dstack",
            vec![list(vec![load("a"), load("b")])],
        ))],
    )
    .expect("np.dstack is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(1), Some(3), Some(2)]);
}

#[test]
fn dstack_promotes_a_rank_two_operand_by_appending() {
    let result = run(
        vec![array_input("a", &[2, 2]), array_input("b", &[2, 2])],
        vec![ret(call(
            Some("np"),
            "dstack",
            vec![list(vec![load("a"), load("b")])],
        ))],
    )
    .expect("np.dstack is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(2), Some(2), Some(2)]);
}

// ---------------------------------------------------------------------------
// 12. `np.concatenate(..., axis=None)` flattens (§5.6)
// ---------------------------------------------------------------------------

#[test]
fn concatenate_with_axis_none_flattens() {
    let result = run(
        vec![array_input("a", &[2, 2]), array_input("b", &[1, 2])],
        vec![ret(call_kw(
            Some("np"),
            "concatenate",
            vec![list(vec![load("a"), load("b")])],
            json!({"axis": {"__class__": "ASTConstantNone"}}),
        ))],
    )
    .expect("np.concatenate is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(6)]);
}

#[test]
fn concatenate_without_an_axis_joins_on_axis_zero() {
    let result = run(
        vec![array_input("a", &[2, 2]), array_input("b", &[1, 2])],
        vec![ret(call(
            Some("np"),
            "concatenate",
            vec![list(vec![load("a"), load("b")])],
        ))],
    )
    .expect("np.concatenate is in scope");
    assert_eq!(delivered_dims(&result), vec![Some(3), Some(2)]);
}

// ---------------------------------------------------------------------------
// 13. `list.remove` records a definedness condition (§5.11)
// ---------------------------------------------------------------------------

#[test]
fn list_remove_records_a_definedness_condition() {
    let result = run(
        vec![scalar_input("v")],
        vec![
            assign(name_target("lst"), list(vec![int(1), int(2), int(3)])),
            json!({"__class__": "ASTExprStatement",
                   "expr": call(Some("lst"), "remove", vec![load("v")])}),
            ret(call(None, "len", vec![load("lst")])),
        ],
    )
    .expect("list.remove is in scope");
    assert!(
        checks(&result)
            .iter()
            .any(|text| text.starts_with("S-ListMethod: remove")),
        "the element must be present, got {:?}",
        checks(&result)
    );
}

// ---------------------------------------------------------------------------
// Consistency finding 3: `dtype=` is honoured (§5.1, §5.8)
// ---------------------------------------------------------------------------

#[test]
fn ones_with_dtype_bool_is_a_mask() {
    // `m = np.ones(len(a), dtype=bool); a[m]` is `E-Filter`, whose length is
    // a count variable, not `E-Take`, whose length is the index array's.
    let result = run(
        vec![array_input("a", &[8])],
        vec![
            assign(
                name_target("m"),
                call_kw(
                    Some("np"),
                    "ones",
                    vec![call(None, "len", vec![load("a")])],
                    json!({"dtype": load("bool")}),
                ),
            ),
            ret(subscript(load("a"), load("m"))),
        ],
    )
    .expect("np.ones with a dtype is in scope");
    let mask = result
        .descriptors
        .iter()
        .find(|entry| entry.name.as_deref() == Some("m"))
        .and_then(|entry| entry.desc.as_array())
        .expect("`m` is an array");
    assert_eq!(mask.dtype, Dtype::Bool);
    assert_eq!(
        delivered_dims(&result),
        vec![None],
        "the filtered length is a count variable, not the literal 8"
    );
}

#[test]
fn an_unreadable_dtype_rejects() {
    let outcome = run(
        vec![array_input("a", &[8])],
        vec![ret(call_kw(
            Some("np"),
            "ones",
            vec![int(3)],
            json!({"dtype": int(7)}),
        ))],
    );
    match outcome {
        Err(Reject::Unsupported { construct }) => assert!(
            construct.contains("dtype"),
            "the message names the keyword, got {}",
            construct
        ),
        other => panic!("expected a rejection, got {:?}", other.err()),
    }
}

// ---------------------------------------------------------------------------
// Consistency finding 7: CONST and PROVE obligations are recorded (§4)
// ---------------------------------------------------------------------------

#[test]
fn a_count_and_a_broadcast_record_their_obligations() {
    // `s = a[a != 0]; return s + b`: the count is a CONST question and the
    // broadcast of `[c]` against `[8]` a PROVE one.
    let result = run(
        vec![array_input("a", &[8]), array_input("b", &[8])],
        vec![
            assign(
                name_target("s"),
                subscript(load("a"), binop("ne", load("a"), int(0))),
            ),
            ret(binop("add", load("s"), load("b"))),
        ],
    )
    .expect("a filtered array is in scope");
    assert!(
        obligations_of(&result, ObligationKind::Const)
            .iter()
            .any(|p| p.contains("count")),
        "the count is a CONST question, got {:?}",
        obligations_of(&result, ObligationKind::Const)
    );
    assert!(
        obligations_of(&result, ObligationKind::Prove)
            .iter()
            .any(|p| p.contains("broadcast")),
        "the broadcast is a PROVE question, got {:?}",
        obligations_of(&result, ObligationKind::Prove)
    );
    assert!(
        !obligations_of(&result, ObligationKind::CanFail).is_empty(),
        "CANFAIL obligations are still recorded"
    );
}

// ---------------------------------------------------------------------------
// Defects 14 and 15: an entry's ψ is the state it was recorded in
// ---------------------------------------------------------------------------

/// The `psi` of the entry Δ records under `key`.
fn psi_of(result: &Phase0Result, key: &str) -> crate::phase0::term::BoolId {
    let entry = result
        .descriptors
        .iter()
        .find(|entry| entry.key.encode() == key)
        .unwrap_or_else(|| panic!("no entry at {}", key));
    match &entry.desc {
        Desc::Scalar(s) => s.psi,
        Desc::Array(a) => a.psi,
        Desc::Composite(c) => match c.elems.first() {
            // §3.3: a composite has no ψ of its own; the condition reaches
            // its elements.
            Some(Desc::Scalar(s)) => s.psi,
            Some(Desc::Array(a)) => a.psi,
            other => panic!("the composite's first element is {:?}", other),
        },
        other => panic!("{} is a {:?}, which has no ψ", key, other),
    }
}

/// Defect 14: a use of a name inside a branch runs only under that branch,
/// whatever the ψ of the value it names.
#[test]
fn a_name_used_inside_a_branch_is_recorded_under_the_branch() {
    // if k > 0: y = a  else: y = a
    let result = run(
        vec![array_input("a", &[4]), scalar_input("k")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("y"), load("a"))],
                vec![assign(name_target("y"), load("a"))],
            ),
            ret(load("y")),
        ],
    )
    .expect("both arms bind `y` to the same input");
    // Statement 0 of the `then` arm: expression 0 is the use of `a`.
    let psi = psi_of(&result, "0.then.0|||0");
    assert!(
        !result.terms.is_true(psi),
        "the use of `a` is guarded by the branch, got ψ = {}",
        result.terms.pretty_bool(psi)
    );
    assert!(
        result.terms.is_true(psi_of(&result, "|||0")),
        "the input itself is unguarded"
    );
}

/// Defect 15: the delivery exists when the circuit returned, not under the
/// path condition left after the last return, which excludes it.
#[test]
fn the_delivery_is_recorded_under_the_condition_it_was_returned_at() {
    // if k > 0: return 1 else: return 2
    let result = run(
        vec![scalar_input("k")],
        vec![cond(
            binop("gt", load("k"), int(0)),
            vec![ret(int(1))],
            vec![ret(int(2))],
        )],
    )
    .expect("every path returns a scalar");
    let key = result
        .delivery
        .first()
        .expect("the circuit delivers a value")
        .encode();
    let psi = psi_of(&result, &key);
    assert!(
        !result.terms.is_false(psi),
        "the delivery is claimed on no path, ψ = {}",
        result.terms.pretty_bool(psi)
    );
}

// ---------------------------------------------------------------------------
// Defect 17: `pop` and `remove` leave the elements after the removed one
// ---------------------------------------------------------------------------

/// `v.m(args)` as a statement, the form a list method takes.
fn method_stmt(receiver: &str, member: &str, args: Vec<Value>) -> Value {
    json!({"__class__": "ASTExprStatement",
           "expr": {"__class__": "ASTNamedAttribute", "target": receiver,
                    "member": member, "args": args, "kwargs": {}}})
}

/// The constant value of element `at` of the delivered composite, when it has
/// one.
fn delivered_element(result: &Phase0Result, at: usize) -> Option<i128> {
    let composite = delivered(result).as_composite().expect("a list");
    let term = composite.elems[at]
        .as_scalar()
        .expect("an integer element")
        .term
        .as_int()
        .expect("an integer term");
    result.terms.constant_value(term)
}

/// `pop(i)` with a runtime index: every element after the removed one moves
/// down, so element 0 is no longer the constant it was.
#[test]
fn pop_with_a_runtime_index_shifts_the_elements_after_it() {
    let result = run(
        vec![scalar_input("i")],
        vec![
            assign(name_target("v"), list(vec![int(1), int(2), int(3)])),
            method_stmt("v", "pop", vec![load("i")]),
            ret(load("v")),
        ],
    )
    .expect("a list of three constants");
    let composite = delivered(&result).as_composite().expect("a list");
    assert_eq!(composite.elems.len(), 2, "`pop` drops one element");
    assert_eq!(
        delivered_element(&result, 0),
        None,
        "element 0 is `ite(i <= 0, 2, 1)`, not a constant"
    );
}

/// `pop(i)` with a constant index folds the shift away.
#[test]
fn pop_with_a_constant_index_keeps_the_remaining_elements() {
    let result = run(
        vec![],
        vec![
            assign(name_target("v"), list(vec![int(1), int(2), int(3)])),
            method_stmt("v", "pop", vec![int(0)]),
            ret(load("v")),
        ],
    )
    .expect("a list of three constants");
    assert_eq!(delivered_element(&result, 0), Some(2));
    assert_eq!(delivered_element(&result, 1), Some(3));
}

/// `remove(v)` drops the *first* element equal to `v`, not element 0.
#[test]
fn remove_drops_the_first_matching_element() {
    let result = run(
        vec![],
        vec![
            assign(
                name_target("v"),
                list(vec![int(1), int(2), int(2), int(3)]),
            ),
            method_stmt("v", "remove", vec![int(2)]),
            ret(load("v")),
        ],
    )
    .expect("a list of four constants");
    assert_eq!(delivered_element(&result, 0), Some(1));
    assert_eq!(delivered_element(&result, 1), Some(2));
    assert_eq!(delivered_element(&result, 2), Some(3));
}

/// `remove` on elements that carry no term cannot say which one matched.
#[test]
fn remove_on_elements_without_a_term_is_a_premise_violation() {
    let outcome = run(
        vec![float_input("a", &[2])],
        vec![
            assign(name_target("v"), list(vec![load("a"), load("a")])),
            method_stmt("v", "remove", vec![load("a")]),
            ret(load("v")),
        ],
    );
    assert!(
        matches!(outcome, Err(Reject::Premise { .. })),
        "got {:?}",
        outcome.err()
    );
}

/// `E-Composite-Index` resolves Python's negative index, so `v[-1]` names the
/// last element rather than the fallback the fold happens to end at.
#[test]
fn a_composite_index_resolves_a_negative_runtime_index() {
    // lst = [1, 2, 3]; lst.pop(i); return lst
    // With `i = -3` the popped element is 1, so what is left is [2, 3]: the
    // shift condition is over the resolved index.
    let result = run(
        vec![scalar_input("i")],
        vec![
            assign(name_target("v"), list(vec![int(1), int(2), int(3)])),
            method_stmt("v", "pop", vec![load("i")]),
            ret(load("v")),
        ],
    )
    .expect("a list of three constants");
    let popped = result
        .descriptors
        .iter()
        .find(|entry| entry.key.encode() == "1|||0")
        .expect("the call node's value");
    let term = popped
        .desc
        .as_scalar()
        .expect("an integer element")
        .term
        .as_int()
        .expect("an integer term");
    let pretty = result.terms.pretty(term);
    assert!(
        pretty.contains("(i < 0)"),
        "the index is resolved before it is compared, got {}",
        pretty
    );
}
