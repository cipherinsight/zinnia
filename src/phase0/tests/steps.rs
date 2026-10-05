//! The intermediates and composite elements of `definition.md` §3.3, and the
//! `defcond_div` of §5.2.
//!
//! Each test walks a small circuit and reads the record back: which key an
//! intermediate took, which rules its producer names, which values it read,
//! and what the value it belongs to names after it. The record invariants of
//! `implementation.md` §5a run on every one of these programs through
//! [`super::fixtures::run`], so a key collision or an operand that resolves
//! to nothing is a failure here as well.

use serde_json::{json, Value};

use crate::phase0::desc::ScalarTerm;
use crate::phase0::rules::Rule;
use crate::phase0::source::Producer;
use crate::phase0::{DescriptorEntry, Phase0Result};

use super::fixtures::*;

/// The entry under the canonical key string, which is how a test names an
/// intermediate (`…#s`) without building a [`ValueKey`] by hand.
fn entry<'a>(result: &'a Phase0Result, key: &str) -> &'a DescriptorEntry {
    result
        .descriptors
        .iter()
        .find(|entry| entry.key.encode() == key)
        .unwrap_or_else(|| {
            panic!(
                "Δ has no entry under `{}`; it has {:?}",
                key,
                result
                    .descriptors
                    .iter()
                    .map(|entry| entry.key.encode())
                    .collect::<Vec<_>>()
            )
        })
}

/// The rule names a producer applied, in application order.
fn rules_of(entry: &DescriptorEntry) -> Vec<&'static str> {
    entry.producer.rules().iter().map(|rule| rule.name()).collect()
}

/// The canonical key strings a producer reads.
fn operands_of(entry: &DescriptorEntry) -> Vec<String> {
    entry
        .producer
        .inputs()
        .iter()
        .map(|key| key.encode())
        .collect()
}

// ---------------------------------------------------------------------------
// §3.3, the `E-Stack` a composite operand is converted by
// ---------------------------------------------------------------------------

/// `np.sum([a, b], axis=0)`: the stacked array is `0|||0#0`, an `E-Stack`
/// over the list, and the reduction itself names one rule and reads it.
#[test]
fn a_composite_operand_is_stacked_into_an_intermediate() {
    let result = run(
        vec![array_input("a", &[4]), array_input("b", &[4])],
        vec![ret(call_kw(
            Some("np"),
            "sum",
            vec![json!({"__class__": "ASTSquareBrackets",
                        "values": [load("a"), load("b")]})],
            vec![("axis", int(0))],
        ))],
    )
    .expect("np.sum over a list of two arrays");

    let stacked = entry(&result, "0|||0#0");
    assert_eq!(rules_of(stacked), vec!["E-Stack"]);
    assert_eq!(
        operands_of(stacked),
        vec!["0|||1".to_string()],
        "the stack reads the list"
    );
    let array = stacked.desc.as_array().expect("the stack built an array");
    assert_eq!(array.rank(), 2, "two arrays of rank 1 stack to rank 2");

    let reduction = entry(&result, "0|||0");
    assert_eq!(rules_of(reduction), vec!["E-Reduce-Axis"]);
    assert!(
        operands_of(reduction).contains(&"0|||0#0".to_string()),
        "the reduction reads the stacked array, got {:?}",
        operands_of(reduction)
    );
}

// ---------------------------------------------------------------------------
// §3.3, one rule application per element of a composite output
// ---------------------------------------------------------------------------

/// `np.unravel_index(i, a.shape)` computes each coordinate from the flat
/// index by arithmetic (§5.4), so the tuple keys no elements: every element
/// is a term over the flat index's variable and is named by that term
/// (§3.3, "Elements of a composite").
#[test]
fn unravel_index_keys_no_element_and_names_the_arithmetic() {
    let result = run(
        vec![array_input("a", &[8, 6]), scalar_input("i")],
        vec![ret(call(
            Some("np"),
            "unravel_index",
            vec![load("i"), call(Some("a"), "shape", vec![])],
        ))],
    )
    .expect("np.unravel_index on a rank-2 shape");

    for key in ["0|||0#0", "0|||0#1"] {
        assert!(
            result
                .descriptors
                .iter()
                .all(|entry| entry.key.encode() != key),
            "the coordinates take no key of their own, but Δ has `{}`",
            key
        );
    }
    let tuple = entry(&result, "0|||0");
    let composite = tuple
        .desc
        .as_composite()
        .expect("the value itself is the tuple");
    let terms: Vec<String> = composite
        .elems
        .iter()
        .map(|element| {
            let scalar = element.as_scalar().expect("a coordinate is a scalar");
            let ScalarTerm::Int(t) = scalar.term else {
                panic!("a coordinate carries an integer term");
            };
            result.terms.pretty(t)
        })
        .collect();
    // 8 × 6 in C order: axis 0 is `i div 6`, and needs no `mod 8` because
    // `i < 48` already bounds the quotient; axis 1 is `i mod 6`.
    assert_eq!(terms, vec!["(i div 6)".to_string(), "(i mod 6)".to_string()]);
    assert_eq!(rules_of(tuple), vec!["E-Arith-Data"]);
    assert!(
        operands_of(tuple).contains(&"0|||1".to_string()),
        "the tuple reads the flat index, got {:?}",
        operands_of(tuple)
    );
}

// ---------------------------------------------------------------------------
// §3.3, the elements of a merged composite
// ---------------------------------------------------------------------------

/// `if k > 0: p = (a, b) else: p = (b, a)`: the merge keys each element, so
/// an element that carries no term of its own still has a name.
#[test]
fn a_merged_composite_keys_its_elements() {
    let result = run(
        vec![
            array_input("a", &[4]),
            array_input("b", &[4]),
            scalar_input("k"),
        ],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign(name_target("p"), tuple(vec![load("a"), load("b")]))],
                vec![assign(name_target("p"), tuple(vec![load("b"), load("a")]))],
            ),
            ret(load("p")),
        ],
    )
    .expect("both arms bind a tuple of two arrays");

    // The merge key is the one entry whose producer names a merge rule and
    // whose descriptor is the merged tuple.
    let merge = result
        .descriptors
        .iter()
        .find(|entry| {
            entry.desc.as_composite().is_some()
                && entry
                    .producer
                    .rules()
                    .iter()
                    .any(|rule| matches!(rule, Rule::MergeArray | Rule::MergeScalar))
        })
        .expect("the tuple was merged");
    let merge_key = merge.key.clone();
    for at in 0..2u32 {
        let element = entry(&result, &merge_key.intermediate(at).encode());
        assert_eq!(rules_of(element), rules_of(merge), "element {}", at);
        assert!(
            element.desc.as_array().is_some(),
            "element {} is one of the arrays",
            at
        );
    }
}

// ---------------------------------------------------------------------------
// §6.2, the value of a `range` element and the position of an array element
// ---------------------------------------------------------------------------

/// `for k in range(0, 3): s = s + k`: the value of iteration `j` is an
/// intermediate of the `range` call, so the name the loop binds reads it.
#[test]
fn a_range_element_is_an_intermediate_of_the_range_call() {
    let result = run(
        vec![scalar_input("n")],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "k",
                call(None, "range", vec![int(0), int(3)]),
                vec![assign(
                    name_target("s"),
                    binop("add", load("s"), load("k")),
                )],
            ),
            ret(load("s")),
        ],
    )
    .expect("a counting loop");

    for j in 0..3u32 {
        let element = entry(&result, &format!("1|||0#{}", j));
        assert_eq!(rules_of(element), vec!["E-Const"], "iteration {}", j);
        let scalar = element.desc.as_scalar().expect("a shape scalar");
        let term = scalar.term.as_int().expect("an integer term");
        assert_eq!(
            result.terms.constant_value(term),
            Some(j as i128),
            "iteration {} binds the constant {}",
            j,
            j
        );
    }
    // The name the loop binds reads the element, so it is no longer a
    // binding with an empty operand list.
    let bound = result
        .descriptors
        .iter()
        .find(|entry| entry.name.as_deref() == Some("k"))
        .expect("the loop binds `k`");
    assert!(
        !operands_of(bound).is_empty(),
        "the loop variable names the value it was bound to"
    );
}

/// `for v in a`: each element records the position it was read at, which is
/// the index its `E-Index` used.
#[test]
fn an_iterated_array_element_records_its_position() {
    let result = run(
        vec![array_input("a", &[3])],
        vec![
            assign(name_target("s"), int(0)),
            for_in(
                "v",
                load("a"),
                vec![assign(
                    name_target("s"),
                    binop("add", load("s"), load("v")),
                )],
            ),
            ret(load("s")),
        ],
    )
    .expect("iteration over a fixed-shape array");

    let elements: Vec<&DescriptorEntry> = result
        .descriptors
        .iter()
        .filter(|entry| entry.producer.rules() == [Rule::EIndex])
        .collect();
    assert_eq!(elements.len(), 3, "one element per position");
    for (at, element) in elements.iter().enumerate() {
        let position = entry(&result, &element.key.intermediate(0).encode());
        let scalar = position.desc.as_scalar().expect("the position is a scalar");
        let term = scalar.term.as_int().expect("an integer term");
        assert_eq!(
            result.terms.constant_value(term),
            Some(at as i128),
            "element {} was read at {}",
            at,
            at
        );
        assert!(
            operands_of(element).contains(&position.key.encode()),
            "element {} reads its position",
            at
        );
    }
}

// ---------------------------------------------------------------------------
// §5.2, `defcond_div` on a divisor that carries no term
// ---------------------------------------------------------------------------

/// `x[0] / x[1]` on a Float array: the divisor carries no term, so the
/// condition is a data Boolean, and that Boolean is a value of Δ produced by
/// `E-Cmp` over the divisor.
#[test]
fn a_float_division_records_a_nonzero_comparison() {
    let result = run(
        vec![float_array_input("x", &[2])],
        vec![ret(binop(
            "div",
            subscript(load("x"), int(0)),
            subscript(load("x"), int(1)),
        ))],
    )
    .expect("a Float scalar division");

    let comparison = entry(&result, "0|||0#0");
    assert_eq!(rules_of(comparison), vec!["E-Cmp"]);
    assert_eq!(
        operands_of(comparison),
        vec!["0|||4".to_string()],
        "the comparison reads the divisor"
    );
    let beta = comparison
        .desc
        .as_scalar()
        .expect("a Boolean scalar")
        .term
        .as_bool()
        .expect("a Boolean term");

    let division = entry(&result, "0|||0");
    let defcond = division.desc.defcond().expect("the division has a defcond");
    assert_eq!(defcond, beta, "the division's defcond is the comparison");
    assert!(
        check_provenances(&result)
            .iter()
            .any(|name| name.starts_with("E-Arith-Data")),
        "the condition entered Φ as a check, got {:?}",
        check_provenances(&result)
    );
}

/// The same division with an integer divisor compares the divisor's term
/// with 0 rather than minting a Boolean.
#[test]
fn an_integer_division_compares_the_divisors_term() {
    let result = run(
        vec![scalar_input("n")],
        vec![ret(binop("div", int(1), load("n")))],
    )
    .expect("an integer division");

    let division = entry(&result, "0|||0");
    let defcond = division.desc.defcond().expect("the division has a defcond");
    assert!(
        result.terms.pretty_bool(defcond).contains("n"),
        "the condition names the divisor, got {}",
        result.terms.pretty_bool(defcond)
    );
    assert!(
        result
            .descriptors
            .iter()
            .all(|entry| entry.key.step == 0),
        "no comparison value was minted"
    );
}

// ---------------------------------------------------------------------------
// The receiver of a method call on a name
// ---------------------------------------------------------------------------

/// `a.sum()` reads `a`: the receiver is a name and not an expression the
/// walker evaluated, so the call has to name it itself
/// (`../../phase1/rules/README.md` §10 item 10).
#[test]
fn a_method_call_on_a_parameter_names_its_receiver() {
    let result = run(
        vec![array_input("a", &[4])],
        vec![ret(call(Some("a"), "sum", vec![]))],
    )
    .expect("a.sum()");

    let reduction = entry(&result, "0|||0");
    assert_eq!(rules_of(reduction), vec!["E-Reduce-Scalar"]);
    assert_eq!(
        operands_of(reduction),
        vec!["|||0".to_string()],
        "the reduction reads the input array"
    );
}

// ---------------------------------------------------------------------------
// §5.11, the elements of the composite a list method rebinds
// ---------------------------------------------------------------------------

/// `v = []; v.append(a); v.append(b)`: each version of `v` keys its
/// elements, and an element `append` kept is the value it came from under a
/// new key.
#[test]
fn a_list_method_keys_the_elements_of_the_new_version() {
    let append = |name: &str| {
        json!({"__class__": "ASTExprStatement",
               "expr": call(Some("v"), "append", vec![load(name)])})
    };
    let result = run(
        vec![array_input("a", &[4]), array_input("b", &[4])],
        vec![
            assign(
                name_target("v"),
                json!({"__class__": "ASTSquareBrackets", "values": []}),
            ),
            append("a"),
            append("b"),
            ret(load("v")),
        ],
    )
    .expect("a list built by two appends");

    let version = result
        .descriptors
        .iter()
        .filter(|entry| entry.producer.rules() == [Rule::SListMethod])
        .next_back()
        .expect("the second append rebound `v`")
        .key
        .clone();
    let first = entry(&result, &version.intermediate(0).encode());
    let second = entry(&result, &version.intermediate(1).encode());
    assert!(
        matches!(first.producer, Producer::Alias(_)),
        "the element the second append kept is the first version's"
    );
    assert!(
        matches!(second.producer, Producer::Alias(_)),
        "the element the second append added is the argument"
    );
    assert!(
        first.desc.as_array().is_some() && second.desc.as_array().is_some(),
        "both elements are the arrays that were appended"
    );
}

/// A Float array input, which `fixtures.rs` has no builder for because every
/// other walk test is about shapes and not dtypes.
fn float_array_input(name: &str, shape: &[usize]) -> Value {
    json!({"name": name,
           "annotation": {"dt": {"NDArray": {"shape": shape, "dtype": "Float"}}}})
}
