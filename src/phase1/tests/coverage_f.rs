//! Coverage of the shapes the second sweep rejected in the representation
//! and mapping groups: the diagonal of a matrix, a deletion and an
//! insertion along one axis, a rank-0 segment of a join, a zero-length
//! array, and the run-time inferred axis of a reshape.

use super::fixtures::*;
use crate::phase1::interface::{Interface, Mu, MuForm};
use crate::phase1::rules::RuleId;

/// `np.diag(a)` on a rank-2 `a` is the view `μ_c = (i_0, i_0)`
/// (`mapping/make_ref.md`, the diagonal family), so it costs nothing.
#[test]
fn the_diagonal_of_a_matrix_is_a_view() {
    let program = run(
        vec![float_array_input("a", &[8, 8])],
        vec![ret(call(Some("np"), "diag", vec![load("a")]))],
    )
    .expect("the program is accepted");

    let refs = program.nodes_of(RuleId::MakeRef);
    assert!(!refs.is_empty(), "{}", program.dump());
    let diagonal = refs
        .iter()
        .filter_map(|node| {
            node.outputs.iter().find_map(|&at| {
                match &program.phase1.graph.node(at).id.form {
                    crate::phase1::graph::Form::Represent {
                        interface: Interface::Ref(reference),
                        ..
                    } if reference.map == vec![Mu::Coord(0), Mu::Coord(0)] => {
                        Some(reference.clone())
                    }
                    _ => None,
                }
            })
        })
        .next()
        .unwrap_or_else(|| panic!("a diagonal Ref\n{}", program.dump()));
    assert_eq!(diagonal.form, MuForm::Coordinate { strides: vec![8, 1] });
    assert_eq!(diagonal.ext.len(), 1);
    assert!(program.phase1.violations().is_empty());
}

/// `np.delete(a, 2, axis=1)` at a constant position is a rewiring: every
/// output slot is a fixed slot of the source, so the node emits nothing
/// (`representation/delete.md`, `emitter`).
#[test]
fn a_deletion_at_a_constant_position_is_free() {
    let program = run(
        vec![float_array_input("a", &[3, 3])],
        vec![ret(call_kw(
            Some("np"),
            "delete",
            vec![load("a"), int(2)],
            vec![("axis", int(1))],
        ))],
    )
    .expect("the program is accepted");

    let nodes = program.nodes_of(RuleId::Delete);
    assert!(!nodes.is_empty(), "{}", program.dump());
    for node in &nodes {
        assert_eq!(node.variant, Some("const"));
        assert_eq!(node.parameter("axis"), Some(1));
        assert_eq!(node.parameter("pos"), Some(2));
        assert_eq!(node.parameter("B_0"), Some(3));
        assert_eq!(node.parameter("B_1"), Some(2));
        assert_eq!(node.cost, 0, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `np.insert(a, pos, e)` at a run-time position buys the thermometer and
/// the one-hot of the position and two selects per slot of the axis, and the
/// integer element is cast into the Float payload
/// (`representation/insert.md`, `emitter`).
#[test]
fn an_insertion_at_a_runtime_position_costs_a_thermometer_and_two_selects() {
    let program = run(
        vec![
            float_array_input("a", &[8]),
            scalar_input("pos"),
            scalar_input("element"),
        ],
        vec![ret(call(
            Some("np"),
            "insert",
            vec![load("a"), load("pos"), load("element")],
        ))],
    )
    .expect("the program is accepted");

    let nodes = program.nodes_of(RuleId::Insert);
    assert!(!nodes.is_empty(), "{}", program.dump());
    for node in &nodes {
        assert_eq!(node.variant, Some("runtime"));
        assert_eq!(node.parameter("axis"), Some(0));
        assert_eq!(node.parameter("B_0"), Some(9));
        // therm(9) = 17, onehot(9) = 10, 2 · 8 · 1 select_f, one float_cast.
        assert_eq!(node.cost, 17 + 10 + 16 + 1, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// A stack of rank-0 elements: `np.stack((x, y))` over two scalars is a
/// rank-1 array whose every segment is one wire
/// (`representation/concat.md`, `source_pattern`).
#[test]
fn a_stack_of_rank_zero_segments_wires_one_element_each() {
    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![ret(call(
            Some("np"),
            "stack",
            vec![tuple(vec![load("x"), load("y")])],
        ))],
    )
    .expect("the program is accepted");
    let nodes = program.nodes_of(RuleId::Concat);
    assert!(!nodes.is_empty(), "{}", program.dump());
    for node in &nodes {
        assert_eq!(node.parameter("B"), Some(2));
        assert_eq!(node.cost, 0, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `np.zeros(0)`: a value with no element is a bundle of no slot, and
/// `CONSTANT/empty` answers it whatever produced it
/// (`sources/constant.md`, the `empty` variant).
#[test]
fn a_value_with_no_element_is_a_bundle_of_no_slot() {
    let program = run(
        vec![scalar_input("n")],
        vec![ret(call(Some("np"), "zeros", vec![int(0)]))],
    )
    .expect("the program is accepted");
    let nodes = program.nodes_of(RuleId::Constant);
    assert!(!nodes.is_empty(), "{}", program.dump());
    for node in &nodes {
        assert_eq!(node.variant, Some("empty"));
        assert_eq!(node.parameter("B"), Some(0));
        assert_eq!(node.cost, 0, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `np.reshape(a, (-1, n))` with a run-time `n`: the inferred axis is the
/// variable ℓ of `E-Reshape`, whose wire is the quotient of
/// `divmod_d(size(a), n)` (`scalar_control/len_shape.md`, the `/div`
/// variant). The `@requires` is what bounds the *declared* axis, which is a
/// capacity and not this rule's business.
#[test]
fn the_inferred_axis_of_a_reshape_is_a_division() {
    let program = run_full(
        vec![float_array_input("a", &[20]), scalar_input("n")],
        vec![ret(call(
            Some("np"),
            "reshape",
            vec![load("a"), tuple(vec![int(-1), load("n")])],
        ))],
        vec![requires_between("n", 1, 4)],
    )
    .expect("the program is accepted");

    let divisions: Vec<_> = program
        .nodes_of(RuleId::LenShape)
        .into_iter()
        .filter(|node| node.variant == Some("/div"))
        .collect();
    assert!(!divisions.is_empty(), "{}", program.dump());
    for node in &divisions {
        assert_eq!(node.parameter("axis"), Some(0));
        // One `divmod_d` over a width that covers size(a) = 20: 5 bits.
        assert_eq!(node.parameter("w"), Some(5));
        assert_eq!(node.cost, 1 + (5 + 1) + 5, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `NA.astype(float)`: the dtype argument of a cast is a `Class` operand of
/// the `E-Map`, which names no value and is skipped by kind
/// (`elementwise/README.md`, `source_pattern`). The integer elements reach
/// the Float payload through one `float_cast` each.
#[test]
fn the_dtype_argument_of_a_cast_is_not_an_operand() {
    let program = run(
        vec![array_input("na", &[5])],
        vec![ret(method(load("na"), "astype", vec![load("float")]))],
    )
    .expect("the program is accepted");

    let maps = program.nodes_of(RuleId::MapAligned);
    assert!(!maps.is_empty(), "{}", program.dump());
    for node in &maps {
        // One `float_cast` per slot of the capacity box.
        assert_eq!(node.cost, 5, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `(x == a).nonzero()[0]`: an `E-Composite-Index` at a literal position
/// computes nothing, so it has the rows of the element it names
/// (`../build.rs`, `composite_index_delegation`) and `NONZERO` answers them.
#[test]
fn a_composite_index_has_the_rows_of_its_element() {
    let program = run(
        vec![array_input("x", &[9]), scalar_input("a")],
        vec![ret(subscript(
            call(Some("np"), "nonzero", vec![binop("eq", load("x"), load("a"))]),
            vec![int(0)],
        ))],
    )
    .expect("the program is accepted");
    assert!(!program.nodes_of(RuleId::Nonzero).is_empty(), "{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// `np.ravel_multi_index(index, dims)` with a run-time `dims`: phase 0
/// records an `E-Arith-Data` over the two arrays with a fresh data variable,
/// and `SCALAR_OP`'s `/ravel` row is the arithmetic itself
/// (`scalar_control/scalar_op.md`).
#[test]
fn a_ravel_multi_index_over_runtime_extents_is_a_stride_sum() {
    let program = run(
        vec![array_input("index", &[5]), array_input("dims", &[5])],
        vec![ret(call(
            Some("np"),
            "ravel_multi_index",
            vec![load("index"), load("dims")],
        ))],
    )
    .expect("the program is accepted");

    let nodes: Vec<_> = program
        .nodes_of(RuleId::ScalarOp)
        .into_iter()
        .filter(|node| node.variant == Some("/ravel"))
        .collect();
    assert!(!nodes.is_empty(), "{}", program.dump());
    for node in &nodes {
        assert_eq!(node.parameter("r"), Some(5));
        // r − 2 running strides and r − 1 terms of the sum.
        assert_eq!(node.cost, 3 + 4, "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}
