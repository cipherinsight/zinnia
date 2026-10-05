//! Coverage of the roots and the sources over the shapes the second sweep
//! rejected: a check recorded after an early `return`, and the elements of a
//! composite parameter.

use super::fixtures::*;
use crate::phase1::graph::{EnforceTarget, Form};
use crate::phase1::rules::RuleId;

/// How many `Enforce` roots the graph holds.
fn enforce_roots(program: &Program) -> usize {
    program
        .phase1
        .graph
        .roots
        .iter()
        .filter(|&&at| {
            matches!(
                program.phase1.graph.node(at).id.form,
                Form::Enforce { target: EnforceTarget::Check { .. } }
            )
        })
        .count()
}

/// An early `return` gives every later value a ψ of `¬(N ≤ 0)`, so a check
/// recorded after it mentions values the program computes only on that path.
/// The root is raised under that ψ and `ENFORCE_CHECK` gates the constraint
/// by its wire (`enforce_check.md`, `required_input_ports`); under `true` the
/// root would ask for a wire the program does not have there, and phase 1
/// rejected with `child_condition`.
#[test]
fn a_check_after_an_early_return_is_enforced_under_its_own_condition() {
    let program = run_full(
        vec![
            scalar_input("N"),
            varlen_input("a", "N", 8),
            scalar_input("k"),
        ],
        vec![
            cond(
                binop("lte", load("N"), int(0)),
                vec![ret(int(0))],
                vec![],
            ),
            assign("j", binop("add", load("k"), int(1))),
            ret(subscript(load("a"), vec![load("j")])),
        ],
        vec![requires_between("N", 0, 8)],
    )
    .expect("the program is accepted");

    assert!(enforce_roots(&program) > 0, "{}", program.dump());
    let gated: Vec<_> = program
        .nodes_of(RuleId::EnforceCheck)
        .into_iter()
        .filter(|node| node.inputs.len() >= 3)
        .collect();
    assert!(
        !gated.is_empty(),
        "the check carries a gate port beside φ and defcond: {}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// A check outside every branch has ψ = true, so it carries no gate and the
/// constraint is the two-term one.
#[test]
fn a_check_outside_every_branch_carries_no_gate() {
    let program = run_full(
        vec![varlen_input("a", "N", 8), scalar_input("k")],
        vec![ret(subscript(load("a"), vec![load("k")]))],
        Vec::new(),
    )
    .expect("the program is accepted");

    for node in program.nodes_of(RuleId::EnforceCheck) {
        assert!(
            node.inputs.len() <= 2,
            "no gate port without a branch: {:?}",
            node.inputs
        );
    }
    assert!(program.phase1.violations().is_empty());
}

/// Each element of a tuple parameter is a parameter of its own
/// (`../phase0/definition.md` §3.3), so `INPUT_SCALAR` answers it and a
/// comparison between two of them is `SCALAR_OP`'s, which is what
/// `benchmarking/pythran_extrema` needs.
#[test]
fn an_element_of_a_tuple_parameter_is_an_input_scalar() {
    let program = run(
        vec![tuple_input("p", &["Integer", "Integer"])],
        vec![
            assign_tuple(&["lo", "hi"], load("p")),
            ret(binop("lt", load("lo"), load("hi"))),
        ],
    )
    .expect("the program is accepted");

    assert!(
        !program.nodes_of(RuleId::InputScalar).is_empty(),
        "the elements are input scalars: {}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::ScalarOp).is_empty(),
        "the comparison is SCALAR_OP's: {}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}
