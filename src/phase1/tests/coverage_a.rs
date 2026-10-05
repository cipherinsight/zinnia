//! Coverage of the mapping and reading groups over the shapes the step-3
//! sweep rejected: a subscript with several advanced entries, a view call
//! written as a method, a chain of four views, and the required ports a view
//! and a dense bundle raise for a run-time extent.

use super::fixtures::*;
use crate::phase1::graph::{Form, ImplNode, Target};
use crate::phase1::interface::{Extent, Interface};
use crate::phase1::rules::RuleId;

/// The terms a node holds a `Represent(t, Wire)` input for.
fn wired_terms(program: &Program, node: &ImplNode) -> Vec<crate::phase0::term::TermId> {
    node.inputs
        .iter()
        .filter_map(|&at| match &program.phase1.graph.node(at).id.form {
            Form::Represent { target: Target::Term(term), .. } => Some(*term),
            _ => None,
        })
        .collect()
}

/// `b[i, j]` with two index arrays: `GATHER` fetches the elements of *both*,
/// so the node matches both index values and pays the normalisation of each
/// (`gather.md`, `source_pattern` and `emitter`: `Σ_m Q_m · (w_I(m) + 2)`).
///
/// Before this, the rule read the first index array alone and the second
/// entry of the subscript went unread.
#[test]
fn a_subscript_with_two_advanced_entries_reads_both() {
    let program = run(
        vec![
            array_input("b", &[3, 3]),
            array_input("i", &[3, 3]),
            array_input("j", &[3, 3]),
        ],
        vec![ret(subscript(load("b"), vec![load("i"), load("j")]))],
    )
    .expect("the program is accepted");

    let gathers = program.nodes_of(RuleId::Gather);
    assert!(!gathers.is_empty(), "{}", program.dump());
    for node in &gathers {
        // matched: the gathered value, the source and one key per entry.
        assert_eq!(node.matched.len(), 4, "{:?}", node.matched);
        assert_eq!(node.parameter("Q"), Some(9));
        // Two index arrays, nine entries each, one `cmp_64` and one
        // `select` per entry.
        assert_eq!(node.cost, 2 * 9 * (64 + 1 + 1), "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// One advanced entry beside a slice keeps the one-entry shape: the block is
/// the leading axis of the output and the second axis is the slice's.
#[test]
fn one_advanced_entry_beside_a_slice_reads_one_index_array() {
    let program = run(
        vec![array_input("b", &[3, 4]), array_input("i", &[3])],
        vec![ret(subscript(
            load("b"),
            vec![load("i"), slice_range(None, None, None)],
        ))],
    )
    .expect("the program is accepted");

    let gathers = program.nodes_of(RuleId::Gather);
    assert!(!gathers.is_empty(), "{}", program.dump());
    for node in &gathers {
        assert_eq!(node.matched.len(), 3, "{:?}", node.matched);
        assert_eq!(node.parameter("Q"), Some(12));
        // Three index elements, read once each and used on four entries.
        assert_eq!(node.cost, 3 * (64 + 1 + 1), "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// `b[:, I]` puts the index block at the axis the advanced entry sits on,
/// so the index elements are read at the *second* axis of the output batch
/// and the normalisation compares against the extent of source axis 1
/// (`gather.md`, `source_pattern`).
#[test]
fn an_advanced_entry_after_a_slice_addresses_its_own_axis() {
    let program = run(
        vec![array_input("b", &[3, 4]), array_input("i", &[5])],
        vec![ret(subscript(
            load("b"),
            vec![slice_range(None, None, None), load("i")],
        ))],
    )
    .expect("the program is accepted");

    let gathers = program.nodes_of(RuleId::Gather);
    assert!(!gathers.is_empty(), "{}", program.dump());
    for node in &gathers {
        // The output is 3 × 5; the block is axis 1, so the five index
        // elements are read once each.
        assert_eq!(node.parameter("Q"), Some(15));
        assert_eq!(node.cost, 5 * (64 + 1 + 1), "{:?}", node.primitives);
    }
    assert!(program.phase1.violations().is_empty());
}

/// A view whose extent is a run-time term states it on a wire, so the node
/// holds the term requirement that answers it (`mapping/README.md` §2.4,
/// `make_ref.md`'s `ext_k` port). A bundle that named an extent with no
/// provider would state a wire phase 3 cannot emit.
#[test]
fn a_view_holds_a_port_for_every_runtime_extent() {
    let program = run_full(
        vec![array_input("a", &[8]), scalar_input("i")],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![subscript(load("a"), vec![slice_range(Some(load("i")), None, None)])],
        ))],
        vec![requires_between("i", 0, 8)],
    )
    .expect("the program is accepted");

    let refs: Vec<&ImplNode> = program
        .nodes_of(RuleId::MakeRef)
        .into_iter()
        .chain(program.nodes_of(RuleId::MakeRefRuntime))
        .collect();
    assert!(!refs.is_empty(), "{}", program.dump());
    let mut seen = false;
    for node in &refs {
        let terms = wired_terms(&program, node);
        for output in &node.outputs {
            let Form::Represent { interface: Interface::Ref(reference), .. } =
                &program.phase1.graph.node(*output).id.form
            else {
                continue;
            };
            for extent in &reference.ext {
                let Extent::Term(term) = extent else { continue };
                seen = true;
                assert!(
                    terms.contains(term),
                    "the `Ref` states a run-time extent with no port: {:?}\n{}",
                    reference.ext,
                    program.dump()
                );
            }
        }
    }
    assert!(seen, "the fixture has a view with a run-time extent");
}

/// The same for `BUILD_DENSE`: an axis on which CONST failed is a
/// `shape_k` port, and the node is not built without it
/// (`build_dense.md`, `required_input_ports`).
#[test]
fn a_dense_bundle_holds_a_port_for_every_runtime_extent() {
    let program = run_full(
        vec![array_input("a", &[8]), scalar_input("i")],
        vec![ret(binop(
            "mul",
            subscript(load("a"), vec![slice_range(Some(load("i")), None, None)]),
            int(2),
        ))],
        vec![requires_between("i", 0, 8)],
    )
    .expect("the program is accepted");

    let dense = program.nodes_of(RuleId::BuildDense);
    assert!(!dense.is_empty(), "{}", program.dump());
    for node in &dense {
        let terms = wired_terms(&program, node);
        for key in &node.matched {
            let Some(entry) = program.phase0.entry(key) else { continue };
            let Some(array) = entry.desc.as_array() else { continue };
            for &dimension in &array.dims {
                if program.phase0.terms.constant_value(dimension).is_some() {
                    continue;
                }
                assert!(
                    terms.contains(&dimension),
                    "the bundle states a run-time extent with no port\n{}",
                    program.dump()
                );
            }
        }
    }
}

/// A view call written as a method reads its parameters after the receiver.
///
/// `np.swapaxes(a, 1, 2)` writes the array as argument 0, so the two axes are
/// arguments 1 and 2; `a.swapaxes(1, 2)` writes the array as the receiver and
/// the axes are arguments 0 and 1. Reading the method spelling at the module
/// spelling's positions gave `swapaxes` and `moveaxis` no permutation at all
/// and `transpose` the default reversal (`mapping/README.md` §3,
/// `make_ref.md`'s `source_pattern`).
#[test]
fn a_view_written_as_a_method_reads_its_own_parameters() {
    let permutations = |program: &Program| -> Vec<Vec<usize>> {
        program
            .phase1
            .graph
            .requirements
            .iter()
            .filter_map(|node| match &node.id.form {
                Form::Represent { interface: Interface::Ref(reference), .. } => {
                    Some(reference.map.clone())
                }
                _ => None,
            })
            .map(|map| {
                map.iter()
                    .map(|expression| match expression {
                        crate::phase1::interface::Mu::Coord(axis) => *axis,
                        _ => usize::MAX,
                    })
                    .collect()
            })
            .collect()
    };

    // `a.swapaxes(0, 1)` over a rank-3 input: μ_c = (i_1, i_0, i_2).
    let method_form = run(
        vec![array_input("a", &[2, 3, 4])],
        vec![ret(method(load("a"), "swapaxes", vec![int(0), int(1)]))],
    )
    .expect("the program is accepted");
    assert!(
        permutations(&method_form).contains(&vec![1, 0, 2]),
        "{:?}",
        permutations(&method_form)
    );

    // The module spelling of the same view is the same map.
    let module_form = run(
        vec![array_input("a", &[2, 3, 4])],
        vec![ret(call(
            Some("np"),
            "swapaxes",
            vec![load("a"), int(0), int(1)],
        ))],
    )
    .expect("the program is accepted");
    assert!(
        permutations(&module_form).contains(&vec![1, 0, 2]),
        "{:?}",
        permutations(&module_form)
    );
}

/// A chain of four views over one input: each is a `Ref` over the one
/// before, `REINDEX_REF` flattens it, and the whole chain is materialised
/// once at the end (`reindex_ref.md`; `mapping/README.md` §2.3, span).
#[test]
fn a_chain_of_four_views_is_one_flat_map() {
    let program = run(
        vec![array_input("a", &[8, 6])],
        vec![ret(method(
            method(
                method(
                    method(load("a"), "reshape", vec![int(4), int(2), int(3), int(2)]),
                    "swapaxes",
                    vec![int(1), int(2)],
                ),
                "transpose",
                vec![int(1), int(0), int(2), int(3)],
            ),
            "reshape",
            vec![int(-1), int(2), int(2)],
        ))],
    )
    .expect("the program is accepted");

    assert!(!program.nodes_of(RuleId::ReindexRef).is_empty(), "{}", program.dump());
    assert!(!program.nodes_of(RuleId::BuildDense).is_empty(), "{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}
