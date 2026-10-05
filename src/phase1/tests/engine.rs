//! The construction of `revision/phase1/README.md` §4, on small programs.

use super::fixtures::*;
use crate::phase1::graph::Form;
use crate::phase1::oracle::questions::Kind;
use crate::phase1::rules::RuleId;

/// The worked example of the brief: `assert x < y; return x + y` on two
/// integer inputs.
///
/// §2 predicts the shape: an `Export` root over `Represent(x + y, Wire)`,
/// which `SCALAR_OP` answers from the two input wires, and an `Enforce(k)`
/// root over the wire of `x < y`. The comparison is a value of Δ, so its
/// term's row is that value's row (`rules/README.md` §3.2) and `SCALAR_OP`
/// answers it rather than `TERM_TO_WIRES`.
#[test]
fn worked_example_has_the_predicted_shape() {
    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![
            assert_stmt(binop("lt", load("x"), load("y"))),
            ret(binop("add", load("x"), load("y"))),
        ],
    )
    .expect("the program is accepted");

    assert_eq!(
        program.rules(),
        vec![
            "ENFORCE_CHECK",
            "EXPORT",
            "INPUT_SCALAR",
            "INPUT_SCALAR",
            "SCALAR_OP",
            "SCALAR_OP",
        ]
    );
    // Two roots: the delivery and the one check.
    assert_eq!(program.phase1.graph.roots.len(), 2);
    let forms: Vec<&Form> = program
        .phase1
        .graph
        .roots
        .iter()
        .map(|&at| &program.phase1.graph.node(at).id.form)
        .collect();
    assert!(forms.iter().any(|form| matches!(form, Form::Represent { .. })));
    assert!(forms.iter().any(|form| matches!(form, Form::Enforce { .. })));

    // The two inputs are shared: one `INPUT_SCALAR` node each, and both the
    // sum and the comparison point at them.
    let inputs = program.nodes_of(RuleId::InputScalar);
    assert_eq!(inputs.len(), 2);
    for node in program.nodes_of(RuleId::ScalarOp) {
        assert_eq!(node.inputs.len(), 2);
    }

    // A holds exactly one CANFAIL, and no width came from the solver.
    assert_eq!(program.answers_of(Kind::CanFail).len(), 1);
    assert_eq!(program.phase1.stats.kind(Kind::Max).asked, 0);
    assert!(program.phase1.violations().is_empty());
}

/// The invariant checker passes on the fixtures and fails on a graph whose
/// epoch discipline is broken by hand.
#[test]
fn the_invariant_checker_sees_a_broken_epoch() {
    let mut program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![
            assert_stmt(binop("lt", load("x"), load("y"))),
            ret(binop("add", load("x"), load("y"))),
        ],
    )
    .expect("the program is accepted");
    assert!(program.phase1.violations().is_empty());

    // Raise the epoch of every node below the `Enforce` root past the check
    // it verifies: that is exactly the fourth invariant of §2.
    for node in &mut program.phase1.graph.implementations {
        if node.rule != RuleId::EnforceCheck {
            node.epoch = 99;
        }
    }
    let violations = program.phase1.violations();
    assert!(
        violations
            .iter()
            .any(|violation| violation.invariant.contains("no check is verified")),
        "{:?}",
        violations
    );
}

/// Two values of Δ that carry one hash-consed term are one row, so the term
/// is asked once and the wire is one requirement with one provider
/// (`rules/README.md` §3.2).
#[test]
fn one_term_asked_twice_is_one_row() {
    let program = run(
        vec![varlen_input("x", "N", 16)],
        vec![
            assign("n", call(None, "len", vec![load("x")])),
            assign("b", binop("add", load("n"), int(1))),
            assign("c", binop("add", load("n"), int(1))),
            ret(binop("add", load("b"), load("c"))),
        ],
    )
    .expect("the program is accepted");

    // `b` and `c` have the term `n_N + 1`, hash-consed to one node, so one
    // `SCALAR_OP` answers both; the sum is the second.
    let scalars = program.nodes_of(RuleId::ScalarOp);
    assert_eq!(scalars.len(), 2, "{}", program.dump());
    let sum = scalars
        .iter()
        .find(|node| node.inputs.len() == 2)
        .expect("the sum reads two wires");
    assert_eq!(sum.inputs[0], sum.inputs[1], "both sides are the same wire");

    // A holds one answer per distinct question: no two entries share a key,
    // and the count is what the statistics say was asked minus the memo
    // hits.
    let mut keys: Vec<_> = program
        .phase1
        .answers
        .entries()
        .iter()
        .map(|answer| answer.key.clone())
        .collect();
    let asked = keys.len();
    keys.dedup();
    assert_eq!(keys.len(), asked, "A holds a question twice");
    let memo: u64 = program
        .phase1
        .stats
        .kinds
        .values()
        .map(|counts| counts.memo_hits)
        .sum();
    assert_eq!(
        program.phase1.stats.questions() - memo,
        program.phase1.answers.len() as u64
    );
    assert!(program.phase1.violations().is_empty());
}

/// A reassigned name and an `if`/merge both give a DAG: Δ's order is a
/// topological order of the source graph, so §4's expansion never cycles.
#[test]
fn reassignment_and_a_merge_stay_acyclic() {
    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![
            assign("s", binop("add", load("x"), int(1))),
            assign("s", binop("add", load("s"), int(1))),
            assign("s", binop("add", load("s"), int(1))),
            cond(
                binop("lt", load("x"), load("y")),
                vec![assign("s", binop("add", load("s"), load("y")))],
                vec![assign("s", binop("sub", load("s"), load("y")))],
            ),
            ret(load("s")),
        ],
    )
    .expect("the program is accepted");

    assert!(program.phase1.violations().is_empty());
    assert!(
        !program.nodes_of(RuleId::Select).is_empty(),
        "the merge at the end of the `if` is a SELECT"
    );
}

/// A check identical to an earlier one is implied by it, so its CANFAIL
/// answer is "no" and it needs no root; the node the surviving root reads
/// has an epoch below the check's position.
///
/// The two asserts are over a *shape* scalar, `len(x)`: phase 0 models a
/// comparison of shape scalars as a Boolean term over Φ's variables, so the
/// second check is a consequence of the first. A comparison of data scalars
/// would be an opaque Boolean and the second check would be independent.
#[test]
fn a_repeated_check_is_implied_and_epochs_stay_below_it() {
    let program = run(
        vec![varlen_input("x", "N", 16)],
        vec![
            assign("n", call(None, "len", vec![load("x")])),
            assert_stmt(binop("lte", load("n"), int(8))),
            assert_stmt(binop("lte", load("n"), int(8))),
            ret(load("n")),
        ],
    )
    .expect("the program is accepted");

    assert_eq!(program.phase0.phi.check_count(), 2, "phase 0 recorded both asserts");
    let canfail = program.answers_of(Kind::CanFail);
    assert_eq!(canfail.len(), 2, "one CANFAIL per check");
    let enforced = program
        .phase1
        .graph
        .roots
        .iter()
        .filter(|&&at| matches!(program.phase1.graph.node(at).id.form, Form::Enforce { .. }))
        .count();
    assert_eq!(enforced, 1, "the second check is implied by the first");

    // The surviving root carries the epoch filter of `rules/README.md` §6.2,
    // and every node below it satisfies it.
    let root = *program
        .phase1
        .graph
        .roots
        .iter()
        .find(|&&at| matches!(program.phase1.graph.node(at).id.form, Form::Enforce { .. }))
        .expect("an Enforce root");
    let position = program.phase1.graph.node(root).id.filter.epoch_below;
    assert_eq!(position, Some(1));
    assert!(program.phase1.violations().is_empty());
}

/// The dominance rule of §4: equal outputs and equal children are compared
/// and the costlier dropped; different children are never compared.
#[test]
fn dominance_compares_only_equal_children() {
    use crate::phase1::build::dominated;
    use crate::phase1::graph::{Graph, ImplId, ImplNode, ReqId};

    let node = |variant, cost: i128, inputs: Vec<ReqId>| ImplNode {
        rule: RuleId::LenShape,
        matched: Vec::new(),
        inputs,
        outputs: Vec::new(),
        parameters: Vec::new(),
        epoch: 0,
        cost,
        primitives: Vec::new(),
        answers: Vec::new(),
        readers: Vec::new(),
        variant: Some(variant),
        dropped: false,
    };
    // Three nodes answering one requirement: two over the same port, one
    // over another. Only the first two are ever compared.
    let mut graph = Graph::default();
    let cheap = graph.add_implementation(node("a", 3, vec![ReqId(0)]));
    let costly = graph.add_implementation(node("b", 9, vec![ReqId(0)]));
    let other = graph.add_implementation(node("c", 9, vec![ReqId(1)]));
    assert_eq!(dominated(&graph, &[cheap, costly, other]), vec![costly]);

    // Equal cost: the later node goes, so the result does not depend on the
    // order the rules ran in.
    let mut graph = Graph::default();
    let first: ImplId = graph.add_implementation(node("a", 3, vec![ReqId(0)]));
    let second = graph.add_implementation(node("b", 3, vec![ReqId(1)]));
    let third = graph.add_implementation(node("c", 3, vec![ReqId(0)]));
    assert_eq!(dominated(&graph, &[first, second, third]), vec![third]);
}

/// The real rows: no two nodes of one row have equal outputs and equal
/// children, because the pruning dropped one of any such pair.
#[test]
fn pruning_leaves_no_equal_pair() {
    // A variable-length input has four `INPUT_STORE` bundles — `bind`,
    // `box`, `packed` and `packed_canon` — of which `packed` and
    // `packed_canon` share one port and differ in cost and in output, so
    // both stay.
    let program = run(
        vec![varlen_input("x", "N", 16)],
        vec![ret(call(None, "len", vec![load("x")]))],
    )
    .expect("the program is accepted");

    let stores = program.nodes_of(RuleId::InputStore);
    assert!(!stores.is_empty());
    for node in &stores {
        for other in &stores {
            if std::ptr::eq(*node, *other) {
                continue;
            }
            assert!(
                node.outputs != other.outputs || node.inputs != other.inputs,
                "an equal pair survived the pruning"
            );
        }
    }
    assert!(program.phase1.violations().is_empty());
}

/// The delivery of a length reads the extent wire of a bundle of the input
/// at no cost, and `LEN_SHAPE/term` offers nothing for a bare variable.
#[test]
fn a_length_reads_a_bundle_extent() {
    let program = run(
        vec![varlen_input("x", "N", 16)],
        vec![ret(call(None, "len", vec![load("x")]))],
    )
    .expect("the program is accepted");

    let lengths = program.nodes_of(RuleId::LenShape);
    assert!(!lengths.is_empty(), "LEN_SHAPE answers `len(x)`");
    assert!(
        lengths.iter().all(|node| node.variant == Some("/read")),
        "a variable with no defining term has no `/term` node"
    );
    assert!(lengths.iter().all(|node| node.cost == 0));
    assert!(program.phase1.violations().is_empty());
}

/// A program whose value needs a rule outside the library is rejected with a
/// reason that names the phase 0 rule and says the library covers it with
/// none: `np.diag` is `E-Diag`, which `rules/README.md` §9 lists as not
/// covered by any group.
#[test]
fn a_missing_provider_names_the_rule() {
    let outcome = run(
        vec![array_input("a", &[4])],
        vec![ret(call(Some("np"), "sum", vec![call(Some("np"), "diag", vec![load("a")])]))],
    );
    let Err(rejection) = outcome else {
        panic!("no rule of the library covers E-Diag, so the program has no provider");
    };
    assert_eq!(rejection.reason(), "no_provider");
    assert!(
        rejection.detail().contains("E-Diag") && rejection.detail().contains("no rule"),
        "{}",
        rejection.detail()
    );
}

/// A `@requires` fact over one parameter narrows the input's interval, so
/// `INPUT_SCALAR` charges the narrower range check; a fact over two
/// parameters is an `Enforce` root of its own (`../README.md` §2).
#[test]
fn a_requires_fact_narrows_an_input_and_roots_a_check() {
    let program = run_full(
        vec![scalar_input("k"), scalar_input("m")],
        vec![ret(binop("add", load("k"), load("m")))],
        vec![
            requires_between("k", 0, 64),
            requires_between("m", 0, 64),
            requires_le("k", "m"),
        ],
    )
    .expect("the program is accepted");

    // ⌈log2 65⌉ = 7 for both parameters, from the `var_range` phase 0
    // recorded — never from a MAX (`rules/README.md` §6.1).
    let inputs = program.nodes_of(RuleId::InputScalar);
    assert_eq!(inputs.len(), 2);
    for node in &inputs {
        assert_eq!(node.parameter("w"), Some(7));
        assert_eq!(node.parameter("a"), Some(0));
        assert_eq!(node.parameter("b"), Some(64));
        // Two range checks, because 65 is not a power of two.
        assert_eq!(node.cost, 14);
    }
    assert_eq!(program.phase1.stats.kind(Kind::Max).asked, 0);

    // `k <= m` mentions two variables, so it is an `Enforce` root with
    // φ = true and no epoch filter of its own.
    let roots: Vec<_> = program
        .phase1
        .graph
        .roots
        .iter()
        .filter(|&&at| matches!(program.phase1.graph.node(at).id.form, Form::Enforce { .. }))
        .collect();
    assert_eq!(roots.len(), 1);
    assert_eq!(
        program.phase1.graph.node(*roots[0]).id.filter.epoch_below,
        None
    );
    assert!(program.phase1.violations().is_empty());
}

/// `Fetch`: the scalar identity of `rules/README.md` §3.2 is the value's wire
/// row, and a batch over an array is the `Values` row the reading rules fill.
#[test]
fn fetch_answers_the_scalar_identity_and_the_reading_rules() {
    use crate::phase0::desc::ValueKey;
    use crate::phase1::build::Builder;
    use crate::phase1::cost::UnitCost;
    use crate::phase1::graph::{EpochFilter, SharingId, Target};
    use crate::phase1::interface::{Batch, Family};
    use crate::phase1::omega::Omega;
    use crate::phase1::oracle::Limits;
    use crate::phase1::table::RowRequest;
    use std::collections::HashMap;

    let ast = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [scalar_input("x"), array_input("a", &[4])],
        "block": [ret(binop("add", load("x"), int(1)))],
        "scalar_requires": [],
    });
    let circuit: crate::ast::ASTCircuit = serde_json::from_value(ast).expect("a valid circuit");
    let chips: HashMap<String, serde_json::Value> = HashMap::new();
    let record = crate::phase0::analyze(
        &circuit,
        &chips,
        &HashMap::new(),
        &crate::phase0::Phase0Config::default(),
    )
    .expect("phase 0 accepts the fixture");

    let omega = Omega::default();
    let cost = UnitCost;
    let mut builder = Builder::new(&record, &circuit, &chips, &omega, &cost, Limits::default());
    let scalar =
        ValueKey { stmt_path: vec![], iters: vec![], calls: vec![], expr_index: 0, step: 0 };
    let array =
        ValueKey { stmt_path: vec![], iters: vec![], calls: vec![], expr_index: 1, step: 0 };
    let request = RowRequest {
        target: Target::Value(scalar.clone()),
        family: Family::Values,
        condition: record.terms.tt(),
        sharing: SharingId::PRIMARY,
        filter: EpochFilter::default(),
        epoch: 0,
        batch: None,
        axes: None,
        extent: None,
        ranks: None,
    };

    // The one-entry batch of a scalar is that scalar's wire.
    let row = builder
        .fetch(&request, &scalar, &Batch::Scalar, record.terms.tt())
        .expect("the scalar identity holds");
    assert_eq!(row.entries.len(), 1);
    assert_eq!(row.entries[0].interface.family(), Family::Wire);

    // A batch over an array is a read: the grid batch of the input's own
    // capacity is answered by `READ_DIRECT` over its box `Store`, whose
    // addresses are the constants of the grid.
    let grid = Batch::grid(&[4]);
    let row = builder
        .fetch(&request, &array, &grid, record.terms.tt())
        .expect("the reading rules answer the batch");
    assert_eq!(row.entries.len(), 1, "one `Values` requirement per batch");
    assert_eq!(row.entries[0].interface.family(), Family::Values);
    let providers = builder.graph.node(row.entries[0].requirement).providers.clone();
    let rules: Vec<&str> = providers
        .iter()
        .map(|&at| builder.graph.implementation(at).rule.as_str())
        .collect();
    assert!(rules.contains(&"READ_DIRECT"), "{:?}", rules);
}

/// A scalar used before and after an assert on it: the node the `Enforce`
/// reads has an epoch below the check's position, and the node built after
/// the check has one at or above it.
#[test]
fn a_scalar_used_across_a_check_keeps_two_epochs() {
    let program = run(
        vec![varlen_input("x", "N", 16)],
        vec![
            assign("n", call(None, "len", vec![load("x")])),
            assign("a", binop("add", load("n"), int(1))),
            assert_stmt(binop("lte", load("n"), int(8))),
            assign("b", binop("add", load("n"), int(2))),
            ret(binop("add", load("a"), load("b"))),
        ],
    )
    .expect("the program is accepted");

    let root = *program
        .phase1
        .graph
        .roots
        .iter()
        .find(|&&at| matches!(program.phase1.graph.node(at).id.form, Form::Enforce { .. }))
        .expect("the assert can fail, so it is a root");
    let position = program
        .phase1
        .graph
        .node(root)
        .id
        .filter
        .epoch_below
        .expect("the root carries the epoch filter");
    assert_eq!(position, 1);

    // Everything below the root assumed no check at or after the assert.
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    let mut below = 0;
    while let Some(at) = stack.pop() {
        if !seen.insert(at) {
            continue;
        }
        for &provider in &program.phase1.graph.node(at).providers {
            let node = program.phase1.graph.implementation(provider);
            assert!(
                node.epoch < position,
                "{} at epoch {}",
                node.rule.as_str(),
                node.epoch
            );
            below += 1;
            stack.extend(node.inputs.iter().copied());
        }
    }
    assert!(below > 0);

    // `b` was recorded after the check, so its node assumed it.
    assert!(
        program
            .phase1
            .graph
            .live_implementations()
            .into_iter()
            .any(|at| program.phase1.graph.implementation(at).epoch >= position),
        "{}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// The emitter counts of `rules/README.md` §7, on the rows of
/// `scalar_op.md`: a wrapping addition on two int64 inputs, and a comparison
/// of the same two.
#[test]
fn the_emitter_counts_follow_the_primitives_of_section_7() {
    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![ret(binop("add", load("x"), load("y")))],
    )
    .expect("the program is accepted");
    let sum = program.nodes_of(RuleId::ScalarOp);
    assert_eq!(sum.len(), 1);
    // `wrap_64(x + y)`: range_64 on the low bits, and a two-bit carry check,
    // because `x + y + 2^63` spans three multiples of 2^64.
    assert_eq!(sum[0].cost, 66);

    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![ret(binop("lt", load("x"), load("y")))],
    )
    .expect("the program is accepted");
    let comparison = program.nodes_of(RuleId::ScalarOp);
    assert_eq!(comparison.len(), 1);
    // `cmp_w` is `w + 1` with `w = W(x − y) = 64`.
    assert_eq!(comparison[0].cost, 65);

    // A shape-origin scalar takes `term_to_wires.md`'s row on its own term:
    // `n div 2` is `divmod_2`, `range_{W(q − lo_q)}(q) + range_{⌈log2 2⌉}(r)`
    // with `q ∈ [0, 512]`.
    let program = run(
        vec![varlen_input("x", "N", 1024)],
        vec![
            assign("n", call(None, "len", vec![load("x")])),
            ret(binop("floor_div", load("n"), int(2))),
        ],
    )
    .expect("the program is accepted");
    let division = program.nodes_of(RuleId::ScalarOp);
    assert_eq!(division.len(), 1);
    assert_eq!(division[0].cost, 11);
}

/// A bounded loop that reassigns two names through a tuple: every iteration
/// is a merge on the loop's guard, and the element of the unpacked tuple is
/// named by the composite's key, so `SELECT` resolves its sides through the
/// arm's term rather than through the operand list.
#[test]
fn a_tuple_unpacking_loop_builds_a_merge_chain() {
    let program = run_full(
        vec![scalar_input("n")],
        vec![
            assign("a", int(1)),
            assign("b", int(1)),
            for_range(
                "n",
                vec![assign_tuple(
                    &["a", "b"],
                    tuple(vec![binop("add", load("a"), load("b")), load("a")]),
                )],
            ),
            ret(load("a")),
        ],
        vec![requires_between("n", 0, 4)],
    )
    .expect("the program is accepted");

    // One `SELECT` per iteration for the name the program returns; the
    // merges of `b`, which nothing reads, are never expanded — phase 1
    // expands from the roots down.
    let merges = program.nodes_of(RuleId::Select);
    assert_eq!(merges.len(), 4, "{}", program.dump());
    // The chain: each merge reads the one before it, so exactly one merge
    // is read by no other merge.
    let read: std::collections::HashSet<_> = merges
        .iter()
        .flat_map(|node| node.inputs.clone())
        .collect();
    let outermost = merges
        .iter()
        .filter(|node| node.outputs.iter().all(|at| !read.contains(at)))
        .count();
    assert_eq!(outermost, 1);
    assert!(program.phase1.violations().is_empty());
    assert_eq!(program.phase1.graph.roots.len(), 1);
}

/// The cut of §4, "Acyclicity by construction": a port that asks for a row
/// that is being filled drops that candidate alone, the row is provisional,
/// and the next consumer fills it again on requirement nodes of its own.
///
/// No real cycle occurs with the nine scalar rules, so the library carries
/// the test-only `CYCLE_PROBE` of `rules/cycle_probe.rs`, which answers the
/// wire of a value named `cycle` by requiring that same wire.
#[test]
fn a_cyclic_candidate_is_cut_and_the_row_is_not_memoised() {
    use crate::phase1::graph::Form;
    use crate::phase1::interface::Family;

    let program = run(
        vec![scalar_input("x"), scalar_input("y")],
        vec![
            assign("cycle", binop("add", load("x"), int(1))),
            // Two consumers, so the row is asked for twice.
            ret(binop("add", binop("mul", load("cycle"), load("y")), load("cycle"))),
        ],
    )
    .expect("the program is accepted");

    // The cyclic candidate is absent and the other candidates are there.
    assert!(
        program.nodes_of(RuleId::CycleProbe).is_empty(),
        "the candidate that reads the row to build it is dropped\n{}",
        program.dump()
    );
    assert!(
        !program.nodes_of(RuleId::ScalarOp).is_empty(),
        "the other candidates of the row are built normally"
    );
    assert!(program.phase1.violations().is_empty());

    // The row was not memoised, so each consumer got a fill of its own with
    // requirement nodes of its own, each carrying that fill's number and
    // each holding the provider the cut did not drop.
    let fills: Vec<_> = program
        .phase1
        .graph
        .requirements
        .iter()
        .filter(|node| node.id.provisional.is_some())
        .collect();
    assert_eq!(fills.len(), 2, "one requirement per fill\n{}", program.dump());
    assert_ne!(
        fills[0].id.provisional, fills[1].id.provisional,
        "each provisional fill numbers its own nodes"
    );
    for node in &fills {
        assert!(
            matches!(node.id.form, Form::Represent { ref interface, .. }
                     if interface.family() == Family::Wire),
            "the provisional rows are the marked value's wire"
        );
        assert_eq!(node.providers.len(), 1, "every fill is a full row but the cut");
    }
}

/// A value bound before a bounded loop and reassigned inside it is read
/// after the loop: every iteration is a merge on the guard, the merge exists
/// at the merge point, and the two sides are raised under their own ψ with
/// no fallback (§4, "Conditions of child requirements").
#[test]
fn a_value_merged_in_a_guarded_iteration_is_read_after_the_loop() {
    let program = run_full(
        vec![scalar_input("n")],
        vec![
            assign("s", int(0)),
            for_range("n", vec![assign("s", binop("add", load("s"), int(1)))]),
            ret(load("s")),
        ],
        vec![requires_between("n", 0, 3)],
    )
    .expect("the program is accepted");

    // One `SELECT` per iteration, chained: each reads the one before it.
    let merges = program.nodes_of(RuleId::Select);
    assert_eq!(merges.len(), 3, "{}", program.dump());
    let read: std::collections::HashSet<_> =
        merges.iter().flat_map(|node| node.inputs.clone()).collect();
    let outermost = merges
        .iter()
        .filter(|node| node.outputs.iter().all(|at| !read.contains(at)))
        .count();
    assert_eq!(outermost, 1, "the delivery reads the last merge");
    assert!(program.phase1.violations().is_empty());
}

/// A name first bound *inside* a guarded iteration and read after the loop
/// has no wire: the merge of the second iteration reads the first
/// iteration's binding, which exists only where that iteration ran, so the
/// premise of §4 fails and phase 1 says which value and which conditions.
#[test]
fn a_name_born_inside_a_guarded_iteration_has_no_provider_after_the_loop() {
    let outcome = run_full(
        vec![scalar_input("n"), scalar_input("x")],
        vec![
            for_range("n", vec![assign("s", binop("add", load("x"), int(1)))]),
            ret(load("s")),
        ],
        vec![requires_between("n", 0, 3)],
    );
    let Err(rejection) = outcome else {
        panic!("the first binding exists only under the first iteration's guard");
    };
    assert_eq!(rejection.reason(), "child_condition", "{}", rejection.detail());
    assert!(
        rejection.detail().contains("ψ"),
        "the message names both conditions, got {}",
        rejection.detail()
    );
}

/// A tuple merged across an `if` and returned: Δ keys the composite and not
/// its elements, so each element is named by its own term and `SELECT`
/// answers it through the element path phase 0 recorded for the variable
/// (`rules/README.md` §2).
#[test]
fn a_merged_tuple_is_delivered_element_by_element() {
    let program = run(
        vec![scalar_input("k"), scalar_input("m")],
        vec![
            cond(
                binop("gt", load("k"), int(0)),
                vec![assign("p", tuple(vec![load("k"), load("m")]))],
                vec![assign("p", tuple(vec![load("m"), load("k")]))],
            ),
            ret(load("p")),
        ],
    )
    .expect("both arms bind a tuple of two integer scalars");

    // One `Represent(·, Export)` root per element, in element order.
    assert_eq!(program.phase1.graph.roots.len(), 2, "{}", program.dump());
    // One `SELECT` per merged element, each reading the condition's wire and
    // one side per arm.
    let merges = program.nodes_of(RuleId::Select);
    assert_eq!(merges.len(), 2, "{}", program.dump());
    for node in &merges {
        assert_eq!(node.inputs.len(), 3, "the condition and the two sides");
    }
    // The two inputs are shared by both elements: each is read by both
    // merges, so `INPUT_SCALAR` is instantiated once per parameter.
    assert_eq!(program.nodes_of(RuleId::InputScalar).len(), 2);
    assert!(program.phase1.violations().is_empty());
}




/// A premise that holds only under the arm's negated condition.
///
/// §4's "Conditions of child requirements" is checked syntactically first
/// and by `PROVE(c ⇒ ψ)` otherwise, and `SELECT` reads its else arm under
/// `ψ_v ∧ ¬b` (`select.md`, `correctness_theorem`). Phase 1 writes no term,
/// so `ψ_v ∧ ¬b` is no node of Φ; the oracle takes the condition as a
/// conjunction of literals instead ([`crate::phase1::oracle::questions::Condition`]),
/// which is what lets the solver assume `¬b`.
///
/// Here `ψ` is `n ≤ 8` and `b` is `n > 8`: two different nodes of Φ, so the
/// syntactic test fails, and `ψ` follows from `¬b` alone. Under `ψ_v` — here
/// `true` — it does not follow, and the premise is refused.
#[test]
fn a_premise_holds_under_a_negated_literal_and_not_without_it() {
    use crate::phase0::term::NodeView;
    use crate::phase1::build::Builder;
    use crate::phase1::cost::UnitCost;
    use crate::phase1::omega::Omega;
    use crate::phase1::oracle::answers::Layer;
    use crate::phase1::build::conjunction;
    use crate::phase1::oracle::Limits;
    use crate::phase1::rules::{Candidate, RuleId};
    use std::collections::HashMap;

    // A program whose Φ holds both comparisons over the input's length.
    let ast = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [varlen_input("x", "N", 16), scalar_input("k")],
        "block": [
            assign("n", call(None, "len", vec![load("x")])),
            cond(binop("gt", load("n"), int(8)),
                 vec![assign("a", int(1))], vec![assign("a", int(2))]),
            cond(binop("lte", load("n"), int(8)),
                 vec![assign("c", int(3))], vec![assign("c", int(4))]),
            ret(binop("add", load("a"), load("c"))),
        ],
        "scalar_requires": [],
    });
    let circuit: crate::ast::ASTCircuit = serde_json::from_value(ast).expect("a valid circuit");
    let chips: HashMap<String, serde_json::Value> = HashMap::new();
    let record = crate::phase0::analyze(
        &circuit,
        &chips,
        &HashMap::new(),
        &crate::phase0::Phase0Config::default(),
    )
    .expect("phase 0 accepts the fixture");

    // `b` is `8 < n` and `ψ` is `n ≤ 8`, as phase 0 wrote them.
    let mut condition = None;
    let mut psi = None;
    for at in 0..record.terms.node_count() {
        let literal_eight = |index: u32| {
            matches!(record.terms.node_at(index as usize), NodeView::Const(8))
        };
        match record.terms.node_at(at) {
            NodeView::Lt(left, _) if literal_eight(left) => {
                condition = record.terms.bool_at(at as u32)
            }
            NodeView::Le(_, right) if literal_eight(right) => {
                psi = record.terms.bool_at(at as u32)
            }
            _ => {}
        }
    }
    let (condition, psi) = (
        condition.expect("`n > 8` is a term of Φ"),
        psi.expect("`n ≤ 8` is a term of Φ"),
    );

    let omega = Omega::default();
    let cost = UnitCost;
    let mut builder = Builder::new(&record, &circuit, &chips, &omega, &cost, Limits::default());
    let mut candidate = Candidate::new(RuleId::Select, Some("/scalar"), 0);

    // Under `ψ_v` alone the premise does not hold, and phase 1 says which
    // value and which conditions.
    let always = builder.terms.tt();
    let refused = builder.child_condition(&mut candidate, always, psi, "the arm");
    assert!(matches!(refused, Err(crate::phase1::Reject::ChildCondition { .. })), "{:?}", refused);

    // Under the built `ψ_v ∧ ¬b` it does, and only the solver sees it.
    let arm = builder.under_arm(always, condition, /* negated = */ true);
    let negated = builder.terms.not(condition);
    assert_eq!(arm, negated, "`true ∧ ¬b` is the term `¬b`");
    let accepted = builder.child_condition(&mut candidate, arm, psi, "the arm");
    assert_eq!(accepted.ok(), Some(psi));
    let proof = builder
        .oracle
        .answers
        .entries()
        .iter()
        .find(|answer| answer.key.condition == arm)
        .expect("the premise was asked under the arm's condition");
    assert_eq!(proof.layer, Layer::Solver);
    assert_eq!(proof.verdict, crate::phase1::oracle::answers::Verdict::Proved);

    // The memo key is the term, so one conjunct set is one condition
    // whichever order it was built in (`build::conjunction`).
    let forward = conjunction(&mut builder.terms, &[psi, negated]);
    let backward = conjunction(&mut builder.terms, &[negated, psi]);
    assert_eq!(forward, backward, "one conjunct set is one condition id");
    assert!(
        forward.index() >= builder.base_nodes,
        "`ψ ∧ ¬b` is a term phase 1 built, above phase 0's node count"
    );
}
