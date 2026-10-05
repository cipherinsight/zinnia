//! The phase-0 half of the second coverage pass over phase 1, one test per
//! record change.
//!
//! Each test walks the smallest program that exercises the change and reads
//! the record back: the terms a rule built, the declarations a product fact
//! carries, the dtype an operator returns, the entries a composite parameter
//! keys, and what the work budget counts. The record invariants of
//! `implementation.md` §5a run on every one of these programs through
//! [`super::fixtures::run`].

use serde_json::{json, Value};

use crate::phase0::desc::{Desc, Dtype, ScalarTerm};
use crate::phase0::phi::ConstraintKind;
use crate::phase0::source::Producer;
use crate::phase0::{Phase0Config, Phase0Result, Reject};

use super::fixtures::*;

/// A fixed-shape float array input.
fn float_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Float"}}}})
}

/// A tuple-typed scalar parameter, as `pythran_extrema` declares one.
fn tuple_input(name: &str, elements: &[&str]) -> Value {
    let elems: Vec<Value> = elements.iter().map(|dt| json!(dt)).collect();
    json!({"name": name, "annotation": {"dt": {"Tuple": {"elements": elems}}}})
}

fn call_kw_json(target: Option<&str>, member: &str, args: Vec<Value>, kwargs: Value) -> Value {
    json!({"__class__": "ASTNamedAttribute", "target": target, "member": member,
           "args": args, "kwargs": kwargs})
}

fn string(text: &str) -> Value {
    json!({"__class__": "ASTConstantString", "value": text})
}

/// The elements of the delivered composite, as pretty-printed terms.
fn delivered_terms(result: &Phase0Result) -> Vec<String> {
    let composite = delivered(result)
        .as_composite()
        .expect("a composite was returned");
    composite
        .elems
        .iter()
        .map(|element| {
            let scalar = element.as_scalar().expect("an element is a scalar");
            match scalar.term {
                ScalarTerm::Int(t) => result.terms.pretty(t),
                ScalarTerm::Bool(b) => result.terms.pretty_bool(b),
                ScalarTerm::Opaque => "opaque".to_string(),
            }
        })
        .collect()
}

/// Every check `Φ` carries, pretty-printed.
fn checks(result: &Phase0Result) -> Vec<String> {
    result
        .phi
        .constraints()
        .iter()
        .filter(|c| matches!(c.kind, ConstraintKind::Check { .. }))
        .map(|c| result.terms.pretty_bool(c.formula))
        .collect()
}

/// Every declaration `Φ` carries, pretty-printed.
fn declarations(result: &Phase0Result) -> Vec<String> {
    result
        .phi
        .constraints()
        .iter()
        .filter(|c| matches!(c.kind, ConstraintKind::Declaration))
        .map(|c| result.terms.pretty_bool(c.formula))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. `np.unravel_index` is arithmetic on the flat index (§5.4)
// ---------------------------------------------------------------------------

/// The coordinates of a C-order unravelling on a 5 × 6 shape, with the check
/// that bounds the flat index.
#[test]
fn unravel_index_is_arithmetic_on_the_flat_index() {
    let result = run(
        vec![float_input("a", &[5, 6]), scalar_input("i")],
        vec![ret(call(
            Some("np"),
            "unravel_index",
            vec![load("i"), call(Some("a"), "shape", vec![])],
        ))],
    )
    .expect("a constant shape");
    assert_eq!(
        delivered_terms(&result),
        vec!["(i div 6)".to_string(), "(i mod 6)".to_string()]
    );
    assert_eq!(
        checks(&result),
        vec!["((0 <= i) && (i < 30))".to_string()],
        "the flat index is bounded by the size"
    );
}

/// `order='F'` varies the first axis fastest, so the strides are the
/// products of the extents *before* each axis.
#[test]
fn unravel_index_reads_the_order_argument() {
    let result = run(
        vec![float_input("a", &[5, 6]), scalar_input("i")],
        vec![ret(call_kw_json(
            Some("np"),
            "unravel_index",
            vec![load("i"), call(Some("a"), "shape", vec![])],
            json!({"order": string("F")}),
        ))],
    )
    .expect("a constant shape");
    assert_eq!(
        delivered_terms(&result),
        vec!["(i mod 5)".to_string(), "(i div 5)".to_string()]
    );
}

/// A shape whose extents are not compile-time constants has no stride in the
/// term language, and the rejection names the axis.
#[test]
fn unravel_index_rejects_a_runtime_shape() {
    let outcome = run(
        vec![array_input("dims", &[2]), scalar_input("i")],
        vec![ret(call(
            Some("np"),
            "unravel_index",
            vec![load("i"), load("dims")],
        ))],
    );
    let Err(Reject::Unsupported { construct }) = outcome else {
        panic!("a runtime shape is rejected, got {:?}", outcome.map(|_| ()));
    };
    assert!(
        construct.contains("np.unravel_index") && construct.contains("axis 0"),
        "the rejection names the operator and the axis, got {}",
        construct
    );
}

// ---------------------------------------------------------------------------
// 2. `np.ravel_multi_index` is the linear form (§5.4)
// ---------------------------------------------------------------------------

/// A tuple of coordinates against a constant `dims`: the flat index is
/// `Σ_k stride_k · index_k` and the check bounds each coordinate.
#[test]
fn ravel_multi_index_is_linear_on_a_constant_shape() {
    let result = run(
        vec![scalar_input("r"), scalar_input("c")],
        vec![ret(call(
            Some("np"),
            "ravel_multi_index",
            vec![
                tuple(vec![load("r"), load("c")]),
                tuple(vec![int(5), int(6)]),
            ],
        ))],
    )
    .expect("a constant dims");
    let scalar = delivered(&result)
        .as_scalar()
        .expect("a scalar was returned");
    let ScalarTerm::Int(t) = scalar.term else {
        panic!("the flat index carries an integer term");
    };
    assert_eq!(result.terms.pretty(t), "((6 * r) + c)");
    assert_eq!(
        checks(&result),
        vec!["(((0 <= r) && (r < 5)) && ((0 <= c) && (c < 6)))".to_string()]
    );
}

// ---------------------------------------------------------------------------
// 3. The linear consequences of a product fact (§3.4)
// ---------------------------------------------------------------------------

/// `np.reshape(a, (-1, n))` on a run-time `n` mints σ with the product fact
/// `σ = ℓ · n`; beside it Φ carries the two declarations §3.4 states, per
/// factor.
#[test]
fn a_product_fact_declares_its_linear_consequences() {
    let result = run(
        vec![float_input("a", &[23]), scalar_input("n")],
        vec![ret(call(
            Some("np"),
            "reshape",
            vec![load("a"), tuple(vec![int(-1), load("n")])],
        ))],
    )
    .expect("a reshape with an inferred axis");
    let declared = declarations(&result);
    assert!(
        declared
            .iter()
            .any(|text| text.contains("(1 <= n)") && text.contains("l_reshape <=")),
        "the factor bound conditioned on the other factor, got {:?}",
        declared
    );
    assert!(
        declared
            .iter()
            .any(|text| text.contains("(l_reshape = 0)")),
        "a zero factor makes the product zero, got {:?}",
        declared
    );
}

// ---------------------------------------------------------------------------
// 4. `deg2rad` and its three siblings return Float (`coverage.md` §5.1)
// ---------------------------------------------------------------------------

/// The front end multiplies by a float constant, so an integer operand is
/// promoted.
#[test]
fn deg2rad_returns_float_on_an_integer_operand() {
    for member in ["deg2rad", "radians", "rad2deg", "degrees"] {
        let result = run(
            vec![scalar_input("degree")],
            vec![ret(call(Some("np"), member, vec![load("degree")]))],
        )
        .unwrap_or_else(|reject| panic!("np.{} is in scope, got {:?}", member, reject));
        let scalar = delivered(&result)
            .as_scalar()
            .unwrap_or_else(|| panic!("np.{} of a scalar is a scalar", member));
        assert_eq!(scalar.dtype, Dtype::Float, "np.{}", member);
    }
}

/// `conj` keeps the operand's dtype, which is what separates it from the four
/// above.
#[test]
fn conjugate_keeps_the_operand_dtype() {
    let result = run(
        vec![scalar_input("x")],
        vec![ret(call(Some("np"), "conj", vec![load("x")]))],
    )
    .expect("np.conj is in scope");
    let scalar = delivered(&result).as_scalar().expect("a scalar");
    assert_eq!(scalar.dtype, Dtype::I64);
}

// ---------------------------------------------------------------------------
// 5. The budget bounds the record as well as the walk (§6.2)
// ---------------------------------------------------------------------------

/// A loop that merges a long list each iteration spends far more record than
/// work, so a budget above the units it walks still stops it. The rejection
/// is the budget's, not the process's.
#[test]
fn the_budget_counts_the_record_and_not_only_the_units() {
    let body = vec![
        assign(name_target("xs"), call(None, "list", vec![load("xs")])),
        cond(
            binop("gt", load("k"), int(0)),
            vec![assign(name_target("xs"), call(None, "list", vec![load("xs")]))],
            vec![],
        ),
    ];
    let program = vec![
        assign(
            name_target("xs"),
            call(Some("np"), "arange", vec![int(64)]),
        ),
        assign(
            name_target("xs"),
            json!({"__class__": "ASTExprAttribute", "target": load("xs"),
                   "member": "tolist", "args": [], "kwargs": {}}),
        ),
        for_in("j", call(None, "range", vec![int(64)]), body),
        ret(load("k")),
    ];
    let inputs = vec![scalar_input("k")];
    let config = Phase0Config {
        work_budget: 5_000,
        ..Phase0Config::default()
    };
    let outcome = run_with(inputs, program, Default::default(), config);
    let Err(Reject::BudgetExceeded { units, budget }) = outcome else {
        panic!("the budget stops it, got {:?}", outcome.map(|_| ()));
    };
    assert_eq!(budget, 5_000);
    assert!(units > budget, "the rejection reports what was spent");
}

// ---------------------------------------------------------------------------
// 7. A composite parameter keys its elements (§3.3)
// ---------------------------------------------------------------------------

/// Every element of a tuple parameter has an entry of Δ with the producer
/// `Input`, which is the only name a `Float` element has: it carries no term.
#[test]
fn a_composite_parameter_keys_its_elements() {
    let result = run(
        vec![tuple_input("a", &["Integer", "Float"])],
        vec![ret(load("a"))],
    )
    .expect("a tuple parameter");
    for (at, dtype) in [(0usize, Dtype::I64), (1, Dtype::Float)] {
        let key = format!("|||0#{}", at);
        let entry = result
            .descriptors
            .iter()
            .find(|entry| entry.key.encode() == key)
            .unwrap_or_else(|| panic!("Δ has an entry under `{}`", key));
        assert!(
            matches!(entry.producer, Producer::Input),
            "element {} is a parameter of its own",
            at
        );
        let Desc::Scalar(scalar) = &entry.desc else {
            panic!("element {} is a scalar", at);
        };
        assert_eq!(scalar.dtype, dtype, "element {}", at);
    }
}

/// An unpacking binds each target to the element's own key, so the binding's
/// operand is the narrowest name the element has.
#[test]
fn unpacking_a_parameter_binds_the_element_key() {
    let result = run(
        vec![tuple_input("a", &["Integer", "Float"])],
        vec![
            json!({"__class__": "ASTAssignStatement",
                   "targets": [{"__class__": "ASTTupleAssignTarget",
                                "targets": [name_target("lo"), name_target("hi")],
                                "star": false}],
                   "value": load("a")}),
            ret(load("hi")),
        ],
    )
    .expect("a tuple parameter unpacked");
    let binding = result
        .descriptors
        .iter()
        .find(|entry| entry.name.as_deref() == Some("hi"))
        .expect("the second target is bound");
    let Producer::Applied { operands, .. } = &binding.producer else {
        panic!("a binding applies S-Assign");
    };
    assert_eq!(
        operands.iter().map(|k| k.encode()).collect::<Vec<_>>(),
        vec!["|||0#1".to_string()],
        "the operand is the element, not the tuple"
    );
}
