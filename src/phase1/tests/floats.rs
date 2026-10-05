//! Float: the rows of `rules/README.md` §7's `fop⟨name⟩` primitive.
//!
//! The library states no gadget for a Float wire — its well-formedness is
//! the backend's encoding (§5) — but it does state which backend instruction
//! each operation denotes, and that is what these tests read off a node: the
//! rule that matched, the `fop` names its emitter counted, and the cost
//! `UnitCost` gives them at one unit each.

use super::fixtures::*;
use crate::phase1::rules::RuleId;

/// The `fop`s of the one node of `rule` that emits any, in emitter order
/// and with its counts. A row is offered once per provider combination, and
/// every such node emits the same instructions, so the first is the row.
fn fops(program: &Program, rule: RuleId) -> Vec<(&'static str, i128)> {
    program
        .nodes_of(rule)
        .iter()
        .map(|node| node.fops())
        .find(|emitted| !emitted.is_empty())
        .unwrap_or_default()
}

/// An elementwise sum over two fixed-shape Float arrays is one `add_f` per
/// slot, and nothing else: no `range_w`, no `wrap_w`.
#[test]
fn a_float_map_is_one_add_f_per_slot() {
    let program = run(
        vec![float_array_input("x", &[4]), float_array_input("y", &[4])],
        vec![ret(binop("add", load("x"), load("y")))],
    )
    .expect("phase 1 accepts a Float elementwise map");

    let nodes = program.nodes_of(RuleId::MapAligned);
    assert!(!nodes.is_empty(), "MAP_ALIGNED has a Float row\n{}", program.dump());
    let node = nodes
        .iter()
        .find(|node| node.fops() == [("add_f", 4)])
        .unwrap_or_else(|| panic!("one `add_f` per slot\n{}", program.dump()));
    // `UnitCost` charges 1 per `fop` (§7), so the four slots cost four.
    assert_eq!(node.cost, 4);
    // A Float wire has no width, so the emitter counts no width gadget.
    assert!(
        node.primitives
            .iter()
            .all(|&(primitive, _)| matches!(primitive, crate::phase1::cost::Primitive::Float(_))),
        "a Float row emits float instructions only: {:?}",
        node.primitives
    );
}

/// `np.isclose(x, y)` is the six float instructions the front end composes
/// it from: `|x − y| ≤ atol + rtol·|y|`.
#[test]
fn isclose_is_six_float_instructions_per_slot() {
    let program = run(
        vec![float_array_input("x", &[2]), float_array_input("y", &[2])],
        vec![ret(call(Some("np"), "isclose", vec![load("x"), load("y")]))],
    )
    .expect("phase 1 accepts `np.isclose` on Float arrays");

    let emitted = fops(&program, RuleId::MapAligned);
    assert_eq!(
        emitted,
        vec![
            ("sub_f", 2),
            ("abs_f", 2),
            ("abs_f", 2),
            ("mul_f", 2),
            ("add_f", 2),
            ("lte_f", 2),
        ],
        "the isclose chain, twice over\n{}",
        program.dump()
    );
    let node = program
        .nodes_of(RuleId::MapAligned)
        .into_iter()
        .find(|node| !node.fops().is_empty())
        .expect("the node that emitted them");
    assert_eq!(node.cost, 12);
}

/// `np.mean` over a Float array: one `add_f` per slot, then one `float_cast`
/// of the fiber's length and one `div_f`, per fiber.
#[test]
fn a_float_mean_divides_once_per_fiber() {
    let program = run(
        vec![float_array_input("x", &[8])],
        vec![ret(call(Some("np"), "mean", vec![load("x")]))],
    )
    .expect("phase 1 accepts a Float mean");

    let nodes = program.nodes_of(RuleId::FoldGroups);
    assert!(!nodes.is_empty(), "FOLD_GROUPS has a `mean` row\n{}", program.dump());
    let node = nodes
        .iter()
        .find(|node| node.fops() == [("add_f", 8), ("float_cast", 1), ("div_f", 1)])
        .unwrap_or_else(|| {
            panic!(
                "eight sum steps and one division: {:?}\n{}",
                nodes.iter().map(|node| node.fops()).collect::<Vec<_>>(),
                program.dump()
            )
        });
    assert_eq!(node.cost, 10);
}

/// A Float scalar read out of an array and divided: the division is
/// `fop⟨div_f⟩` and carries no `divmod` of its own, and the divisor's
/// non-zero test is one `fop⟨ne_f⟩` that `ENFORCE_CHECK` reads.
///
/// `defcond_div` of `../../phase0/definition.md` §5.2 compares a divisor
/// that carries no term with 0 through a data Boolean, and that comparison
/// is the division's intermediate `#0` (§3.3), so `SCALAR_OP` has a value to
/// build the wire on.
#[test]
fn a_float_scalar_division_is_one_div_f() {
    let program = run(
        vec![float_array_input("x", &[2])],
        vec![
            assign("a", subscript(load("x"), vec![int(0)])),
            assign("b", subscript(load("x"), vec![int(1)])),
            ret(binop("div", load("a"), load("b"))),
        ],
    )
    .expect("phase 1 accepts a Float scalar division");

    let rows: Vec<Vec<(&'static str, i128)>> = program
        .nodes_of(RuleId::ScalarOp)
        .iter()
        .map(|node| node.fops())
        .filter(|emitted| !emitted.is_empty())
        .collect();
    assert!(
        rows.contains(&vec![("div_f", 1)]),
        "one `div_f` and nothing else in the division's row\n{}",
        program.dump()
    );
    assert!(
        rows.contains(&vec![("ne_f", 1)]),
        "one `ne_f` for `b != 0`\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::EnforceCheck).is_empty(),
        "the division's `defcond` is enforced\n{}",
        program.dump()
    );
}

/// A cast int → float: `np.mean` over an *integer* array accumulates in the
/// output's dtype, so every slot is read through one `float_cast` before the
/// sum (§2.3, "casts int→float and float→int").
#[test]
fn a_mean_over_integers_casts_every_slot() {
    let program = run(
        vec![array_input("x", &[4])],
        vec![ret(call(Some("np"), "mean", vec![load("x")]))],
    )
    .expect("phase 1 accepts a mean over an integer array");

    assert_eq!(
        fops(&program, RuleId::FoldGroups),
        vec![("float_cast", 4), ("add_f", 4), ("float_cast", 1), ("div_f", 1)],
        "four casts, four sums, one cast of the length and one division\n{}",
        program.dump()
    );
}

/// A check over a Float array's extent reaches its wire.
///
/// The check is `E-Elem`'s index range, whose defcond mentions the array's
/// shape symbol; the wire of that symbol is the extent wire of the array's
/// `Store`, which `INPUT_STORE` provides only because the library now admits
/// a Float element (§2.1). Step 2 rejected the two kernels of this shape
/// with "ENFORCE_CHECK found no wire for the check's terms".
#[test]
fn a_check_over_a_float_arrays_extent_reaches_its_wire() {
    let program = run_full(
        vec![float_varlen_input("x", "N", 16), scalar_input("k")],
        vec![ret(subscript(load("x"), vec![load("k")]))],
        vec![requires_between("k", 0, 3)],
    )
    .expect("phase 1 accepts an element of a Float array");

    assert!(
        !program.nodes_of(RuleId::EnforceCheck).is_empty(),
        "the index check is enforced\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::InputStore).is_empty(),
        "the Float array has a Store, which is where the extent wire comes from\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::LenShape).is_empty(),
        "`LEN_SHAPE` answers the shape symbol the defcond mentions\n{}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// A delivered tuple of Float scalars has a root per element.
///
/// Δ keys the composite and not its elements, and a Float element carries no
/// term, so it has no name through §3.2. The composite is an `E-Composite`
/// with one operand per element, so the element is named by that operand
/// (`build.rs`, `element_target`) and `EXPORT` gives it a cell like any other
/// scalar (§2.3).
#[test]
fn a_delivered_tuple_of_float_scalars_has_a_root_per_element() {
    let program = run(
        vec![float_array_input("x", &[4])],
        vec![
            assign("a", subscript(load("x"), vec![int(0)])),
            assign("b", subscript(load("x"), vec![int(1)])),
            assign("pair", tuple(vec![binop("add", load("a"), load("b")), load("a")])),
            ret(load("pair")),
        ],
    )
    .expect("phase 1 accepts a delivered tuple of Float scalars");

    assert_eq!(program.phase1.graph.roots.len(), 2, "one root per element\n{}", program.dump());
    assert_eq!(
        program
            .nodes_of(RuleId::Export)
            .iter()
            .filter(|node| node.variant == Some("scalar"))
            .count(),
        2,
        "one exported cell per element\n{}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// A Float literal is one `fop⟨constant_float⟩` on a wire.
///
/// It carries no term, so the `/const` variant of `SCALAR_OP` — which
/// substitutes a value CONST decided — does not apply to it; the literal row
/// of §2.3 does, and the AST node is what names it.
#[test]
fn a_float_literal_is_one_constant_float() {
    let program = run(
        vec![float_array_input("x", &[1])],
        vec![
            assign("k", float(2.5)),
            assign("a", subscript(load("x"), vec![int(0)])),
            ret(binop("mul", load("k"), load("a"))),
        ],
    )
    .expect("phase 1 accepts a Float literal");

    assert_eq!(
        fops(&program, RuleId::ScalarOp),
        vec![("constant_float", 1)],
        "the literal's own row; the product is a second node\n{}",
        program.dump()
    );
    assert!(
        program
            .nodes_of(RuleId::ScalarOp)
            .iter()
            .any(|node| node.fops() == [("mul_f", 1)]),
        "and the product is `mul_f`\n{}",
        program.dump()
    );
}
