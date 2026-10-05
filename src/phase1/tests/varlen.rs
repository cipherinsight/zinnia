//! The variable-length rules of step 3, one fixture per rule and one per
//! engine piece they forced: the `Ranks` row of `rules/README.md` §4.5, the
//! projection of `COUNT` onto `PREFIX_RANK`'s wires (§3.4), the later-epoch
//! variant of §6.2, the cut of `revision/phase1/README.md` §4 between
//! `PACK_BY_KEY` and `BUILD_MASKED`, and the `Live::Term` batch that
//! `SETITEM` writes.

use super::fixtures::*;
use crate::phase1::graph::{Form, ImplNode};
use crate::phase1::interface::{Family, Interface, LayoutKind};
use crate::phase1::rules::RuleId;

fn nodes<'a>(program: &'a Program, rule: RuleId) -> Vec<&'a ImplNode> {
    program.nodes_of(rule)
}

/// `m = x > 0`, the Boolean mask every filter fixture selects with. The front
/// end aliases `Boolean` to `Integer` on a parameter, so a mask with dtype
/// Bool comes from a comparison and not from an annotation.
fn mask(name: &str) -> serde_json::Value {
    assign("m", binop("gt", load(name), int(0)))
}

/// `m = x > 0; t = x[m]`, the filter the group's examples use.
fn filtered(name: &str) -> Vec<serde_json::Value> {
    vec![mask(name), assign("t", subscript(load(name), vec![load("m")]))]
}

/// The requirement nodes of one family, with their providers.
fn requirements<'a>(
    program: &'a Program,
    family: Family,
) -> Vec<(&'a Interface, &'a Vec<crate::phase1::graph::ImplId>)> {
    program
        .phase1
        .graph
        .requirements
        .iter()
        .filter_map(|node| match &node.id.form {
            Form::Represent { interface, .. } if interface.family() == family => {
                Some((interface, &node.providers))
            }
            _ => None,
        })
        .collect()
}

/// `PREFIX_RANK` over the box `Store` of a mask: the rank chain is `lin`, so
/// the node costs nothing, and its second bundle is the wire of the count
/// variable, which makes it one node answering two requirements (§3.4,
/// `prefix_rank.md`, `count.md`).
#[test]
fn the_rank_chain_and_the_count_are_one_node() {
    let mut block = filtered("x");
    block.push(ret(call(Some("np"), "sum", vec![load("t")])));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    // One node per ordered bundle of m; the mask's own row holds a box
    // `Store` from `MAP_ALIGNED`, whose validity is constant.
    let ranks = nodes(&program, RuleId::PrefixRank);
    let node = ranks
        .iter()
        .find(|node| node.variant == Some("mask") && node.parameter("B_v") == Some(0))
        .expect("the node over the mask's box `Store`");
    assert_eq!(node.parameter("B"), Some(8));
    assert_eq!(node.parameter("w"), Some(4), "bits(8) = 4");
    assert_eq!(node.parameter("B_v"), Some(0), "the mask's validity is constant");
    assert_eq!(node.cost, 0, "the rank chain is `lin`");

    // §3.4: the `Ranks` bundle and the count's wire on one node.
    assert_eq!(node.outputs.len(), 2, "{}", program.dump());
    let families: Vec<Family> = node
        .outputs
        .iter()
        .filter_map(|&at| program.phase1.graph.node(at).id.form.family())
        .collect();
    assert!(families.contains(&Family::Ranks), "{:?}", families);
    assert!(families.contains(&Family::Wire), "{:?}", families);
    // `COUNT` has no node of its own.
    assert!(nodes(&program, RuleId::Count).is_empty());
    assert!(program.phase1.violations().is_empty());
}

/// A check on the length of a filter reaches the count's wire:
/// `ENFORCE_CHECK` raises the defcond, `TERM_TO_WIRES` compiles `c ≤ 3` from
/// the wire of c, and that wire is `PREFIX_RANK`'s last rank
/// (`count.md`, Example).
#[test]
fn a_check_on_a_count_is_enforced_on_the_rank_chain() {
    let mut block = filtered("x");
    block.push(assert_stmt(binop("lte", call(None, "len", vec![load("t")]), int(3))));
    block.push(ret(call(Some("np"), "sum", vec![load("t")])));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    // The check can fail — c may be 8 — so it is an `Enforce` root.
    let roots: Vec<&Form> = program
        .phase1
        .graph
        .roots
        .iter()
        .map(|&at| &program.phase1.graph.node(at).id.form)
        .collect();
    assert!(
        roots.iter().any(|form| matches!(form, Form::Enforce { .. })),
        "{:?}",
        roots
    );
    // The comparison is compiled by `TERM_TO_WIRES` over the count's wire.
    let terms = nodes(&program, RuleId::TermToWires);
    assert!(
        terms.iter().any(|node| node.cost > 0),
        "the comparison costs a `cmp`\n{}",
        program.dump()
    );
    // The wire of c is a rank chain's last rank, at epoch 0: the check is
    // verified on a node that assumed no check (§6.2).
    let ranks = nodes(&program, RuleId::PrefixRank);
    assert!(!ranks.is_empty(), "{}", program.dump());
    assert!(
        ranks
            .iter()
            .filter(|node| node.variant == Some("mask"))
            .all(|node| node.epoch == 0),
        "the mask is recorded before the check\n{}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// `BUILD_MASKED` keeps the filter in the base's slots: the capacity is the
/// base's, the selection bits are the `Ranks` node's by alias, and the node
/// emits nothing (`build_masked.md`, `filter-aligned`).
#[test]
fn a_filter_is_held_in_place_over_the_bases_slots() {
    let mut block = filtered("x");
    block.push(ret(call(Some("np"), "sum", vec![load("t")])));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    let in_place = nodes(&program, RuleId::BuildMasked);
    assert!(!in_place.is_empty(), "{}", program.dump());
    let aligned = in_place
        .iter()
        .find(|node| node.variant == Some("filter-aligned"))
        .expect("the aligned form, the mask being a box Store over the base's box");
    assert_eq!(aligned.parameter("B"), Some(8), "in place keeps the capacity");
    assert_eq!(aligned.cost, 0, "the validity is the `t` vector by alias");
    // The bundle is ranked and names the mask its ranks came from.
    let masked = requirements(&program, Family::Masked);
    assert!(
        masked.iter().any(|(interface, _)| matches!(interface,
            Interface::Masked { capacity: 8, lambda, .. }
                if matches!(lambda.layout, LayoutKind::Ranked { ranks: Some(_), .. }))),
        "{:?}",
        masked.iter().map(|(interface, _)| interface.render()).collect::<Vec<_>>()
    );
    assert!(program.phase1.violations().is_empty());
}

/// `PACK_BY_KEY` routes the valid positions of the in-place bundle into a
/// payload of the value's own length: `B_in` one-hots over `B` slots and one
/// product per (position, slot) (`pack_by_key.md`, `select`).
#[test]
fn the_packed_form_routes_every_position_by_its_key() {
    let mut block = filtered("x");
    block.push(ret(load("t")));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    let packed = nodes(&program, RuleId::PackByKey);
    assert!(!packed.is_empty(), "{}", program.dump());
    let node = packed
        .iter()
        .find(|node| node.parameter("B") == Some(8))
        .expect("MAX c is the mask's capacity");
    assert_eq!(node.variant, Some("select"));
    assert_eq!(node.parameter("B_in"), Some(8));
    assert_eq!(node.parameter("S"), Some(1));
    // 8 one-hots over 8 slots (9 units each) and 64 products.
    assert_eq!(node.cost, 8 * 9 + 8 * 8);
    // §3.4: the plain and the canonical `Packed` from one node.
    assert_eq!(node.outputs.len(), 2, "{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// A check that bounds the count gives `PACK_BY_KEY` a second node at the
/// check's position, whose MAX is the bound rather than the mask's capacity
/// (`rules/README.md` §6.2).
#[test]
fn a_later_epoch_variant_packs_into_the_checked_bound() {
    let mut block = filtered("x");
    block.push(assert_stmt(binop("lte", call(None, "len", vec![load("t")]), int(3))));
    block.push(ret(load("t")));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    let packed = nodes(&program, RuleId::PackByKey);
    let capacities: Vec<Option<i128>> =
        packed.iter().map(|node| node.parameter("B")).collect();
    assert!(capacities.contains(&Some(3)), "the bound c ≤ 3\n{}", program.dump());
    let tight = packed
        .iter()
        .find(|node| node.parameter("B") == Some(3))
        .expect("the later-epoch node");
    assert!(tight.epoch > 0, "it assumed the check\n{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// `PACK_BY_KEY` over the `Masked` of a filter reads the length of that very
/// filter, whose row `LEN_SHAPE` can answer from the `Packed` this node
/// produces. The cut of §4 drops that candidate rather than closing a cycle,
/// and the packed form is still built.
#[test]
fn the_packed_form_and_the_in_place_form_close_no_cycle() {
    let mut block = filtered("x");
    block.push(ret(load("t")));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    assert!(!nodes(&program, RuleId::PackByKey).is_empty());
    assert!(!nodes(&program, RuleId::BuildMasked).is_empty());
    // The invariant checker walks the requirement → implementation edges and
    // finds no back edge.
    assert!(
        program.phase1.violations().is_empty(),
        "{:?}",
        program.phase1.violations()
    );
}

/// `NONZERO` packs the flat indices of a mask with the routing network of
/// `PACK_BY_KEY`, the keys being the constant slot numbers of a box bundle
/// (`nonzero.md`, `select`).
#[test]
fn nonzero_packs_the_flat_indices_of_the_true_positions() {
    let program = run(
        vec![array_input("x", &[8])],
        vec![
            mask("x"),
            assign("idx", call(Some("np"), "flatnonzero", vec![load("m")])),
            ret(load("idx")),
        ],
    )
    .expect("the program is accepted");

    let indices = nodes(&program, RuleId::Nonzero);
    assert!(!indices.is_empty(), "{}", program.dump());
    let node = indices[0];
    assert_eq!(node.variant, Some("select"));
    assert_eq!(node.parameter("B"), Some(8));
    assert_eq!(node.parameter("B_nz"), Some(8));
    // 8 one-hots over 8 slots; the keys are constants, so no product.
    assert_eq!(node.cost, 8 * 9);
    assert!(program.phase1.violations().is_empty());
}

/// `CONCAT` wires two segments of constant extent: every slot of the result
/// is a fixed slot of one segment, so the node allocates nothing
/// (`concat.md`, `wire`).
#[test]
fn a_constant_concatenation_is_wiring() {
    let program = run(
        vec![array_input("x", &[4]), array_input("y", &[3])],
        vec![
            assign(
                "u",
                call(Some("np"), "concatenate", vec![tuple(vec![load("x"), load("y")])]),
            ),
            ret(call(Some("np"), "sum", vec![load("u")])),
        ],
    )
    .expect("the program is accepted");

    let joined = nodes(&program, RuleId::Concat);
    assert!(!joined.is_empty(), "{}", program.dump());
    let wired = joined
        .iter()
        .find(|node| node.variant == Some("wire"))
        .expect("every extent on the joined axis is constant");
    assert_eq!(wired.parameter("B"), Some(7), "4 + 3 slots");
    assert_eq!(wired.cost, 0, "the flags are the batches' own");
    assert!(program.phase1.violations().is_empty());
}

/// A concatenation one of whose segments has a run-time length is placed by
/// `PACK_SEGMENTS`: the second segment's offset is the first's length, a
/// wire, so it is shifted rather than wired (`pack_segments.md`).
#[test]
fn a_runtime_segment_is_placed_by_a_shifter() {
    let mut block = filtered("x");
    block.push(assign(
        "u",
        call(Some("np"), "concatenate", vec![tuple(vec![load("t"), load("y")])]),
    ));
    block.push(ret(load("u")));
    let program = run(vec![array_input("x", &[8]), array_input("y", &[3])], block)
        .expect("the program is accepted");

    let segments = nodes(&program, RuleId::PackSegments);
    assert!(!segments.is_empty(), "{}", program.dump());
    let node = segments
        .iter()
        .max_by_key(|node| node.parameter("B").unwrap_or(0))
        .expect("one node per combination of the segments' bundles");
    assert_eq!(node.parameter("n"), Some(2));
    assert!(node.parameter("B").unwrap_or(0) >= 3);
    assert!(node.cost > 0, "the second segment is shifted\n{}", program.dump());
    // §3.4: the `Segments` bundle and one `Ref` per segment.
    assert!(node.outputs.len() >= 3, "{}", program.dump());
    // `CONCAT`'s `pack` variant reads it and adds nothing.
    let packed = nodes(&program, RuleId::Concat);
    assert!(
        packed.iter().any(|node| node.variant == Some("pack") && node.cost == 0),
        "{}",
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}

/// `SETITEM` writes one `select` per slot of the old bundle, with the region
/// bits of a slice entry from two thermometers (`setitem.md`, the S row).
#[test]
fn an_in_place_write_costs_one_select_per_slot() {
    let program = run_full(
        vec![array_input("x", &[8]), scalar_input("lo"), scalar_input("hi")],
        vec![
            assign("b", call(Some("np"), "array", vec![load("x")])),
            assign_subscript("b", vec![slice_range(Some(load("lo")), Some(load("hi")), None)], int(0)),
            ret(call(Some("np"), "sum", vec![load("b")])),
        ],
        vec![requires_between("lo", 0, 8), requires_between("hi", 0, 8)],
    )
    .expect("the program is accepted");

    let writes = nodes(&program, RuleId::SetItem);
    assert!(!writes.is_empty(), "{}", program.dump());
    let node = writes[0];
    assert_eq!(node.parameter("Q"), Some(8));
    // Two thermometers over 8 slots (15 each), 8 `and` for the region and 8
    // `select` for the update.
    assert_eq!(node.cost, 2 * 15 + 8 + 8);
    assert!(program.phase1.violations().is_empty());
}

/// `SORT` checks one permutation and `B − 1` adjacent range checks; over an
/// operand of constant extent held in a box the validity column folds away
/// (`sort.md`, the dense fold).
#[test]
fn a_sort_checks_a_permutation_and_the_order() {
    let program = run(
        vec![array_input("x", &[5])],
        vec![assign("s", call(Some("np"), "sort", vec![load("x")])), ret(load("s"))],
    )
    .expect("the program is accepted");

    let sorted = nodes(&program, RuleId::Sort);
    assert!(!sorted.is_empty(), "{}", program.dump());
    let node = sorted
        .iter()
        .find(|node| node.variant == Some("dyn"))
        .expect("a rank-1 operand");
    assert_eq!(node.parameter("B"), Some(5));
    assert_eq!(node.parameter("beta"), Some(64));
    assert_eq!(node.parameter("b"), Some(0), "`SORT` orders on the value alone");
    // `perm` is the backend's and costs 0 here; the order check is
    // (d − 1) range checks of β bits.
    assert_eq!(node.cost, 4 * 64);
    assert!(program.phase1.violations().is_empty());
}

/// `ARGSORT` is `SORT` with the index column: the permutation carries one
/// tuple entry more and the order check is `b = bits(B)` bits wider
/// (`argsort.md`).
#[test]
fn an_argsort_orders_on_the_combined_key() {
    let program = run(
        vec![array_input("x", &[5])],
        vec![assign("r", call(Some("np"), "argsort", vec![load("x")])), ret(load("r"))],
    )
    .expect("the program is accepted");

    let indices = nodes(&program, RuleId::Argsort);
    assert!(!indices.is_empty(), "{}", program.dump());
    let node = indices
        .iter()
        .find(|node| node.variant == Some("dyn"))
        .expect("a rank-1 operand");
    assert_eq!(node.parameter("B"), Some(5));
    assert_eq!(node.parameter("b"), Some(3), "bits(5) = 3");
    assert_eq!(node.cost, 4 * (64 + 3));
    assert!(nodes(&program, RuleId::Sort).is_empty(), "the value is the indices");
    assert!(program.phase1.violations().is_empty());
}

/// `UNIQUE` with `return_inverse` is one node with two values' bundles among
/// its outputs (§3.4), and the inverse of a rank-≥2 operand is keyed, which
/// `READ_KEY_MUX` variant C reads (`unique.md`, `read_key_mux.md`).
#[test]
fn unique_with_the_inverse_is_one_node_with_two_outputs() {
    let program = run(
        vec![array_input("a", &[2, 3])],
        vec![
            assign_tuple(
                &["vals", "idx"],
                call_kw(
                    Some("np"),
                    "unique",
                    vec![load("a")],
                    vec![("return_inverse", json_true())],
                ),
            ),
            ret(call(Some("np"), "sum", vec![load("idx")])),
        ],
    )
    .expect("the program is accepted");

    let distinct = nodes(&program, RuleId::Unique);
    assert!(!distinct.is_empty(), "{}", program.dump());
    let node = distinct[0];
    assert_eq!(node.parameter("B"), Some(6), "the operand's 6 slots");
    // One node, two values: the `Masked` of the distinct elements and the
    // keyed `Store` of the inverse.
    let targets: Vec<String> = node
        .outputs
        .iter()
        .map(|&at| match &program.phase1.graph.node(at).id.form {
            Form::Represent { target, .. } => target.render(),
            other => format!("{:?}", other),
        })
        .collect();
    assert!(
        targets.iter().collect::<std::collections::BTreeSet<_>>().len() >= 2,
        "one node answers both values: {:?}",
        targets
    );
    let stores = requirements(&program, Family::Store);
    assert!(
        stores.iter().any(|(interface, _)| matches!(interface,
            Interface::Store(store) if matches!(store.lambda.layout, LayoutKind::Keyed { .. }))),
        "the inverse of a rank-≥2 operand is keyed"
    );
    assert!(program.phase1.violations().is_empty());
}

/// A keyed owner has no compile-time slot for a key, so a read of it is
/// `READ_KEY_MUX` variant C: one `eq` against every slot's key wire
/// (`read_key_mux.md`), and a fold over it is `EXPAND_FIBERS`' `keyed`
/// variant, the one bundle with `ordered = no`.
#[test]
fn a_keyed_owner_is_read_by_comparison() {
    let program = run(
        vec![array_input("a", &[2, 3])],
        vec![
            assign_tuple(
                &["vals", "idx"],
                call_kw(
                    Some("np"),
                    "unique",
                    vec![load("a")],
                    vec![("return_inverse", json_true())],
                ),
            ),
            ret(subscript(load("idx"), vec![int(0)])),
        ],
    )
    .expect("the program is accepted");

    let muxes = nodes(&program, RuleId::ReadKeyMux);
    assert!(
        muxes.iter().any(|node| node.variant == Some("C")),
        "{}",
        program.dump()
    );
    let node = muxes
        .iter()
        .find(|node| node.variant == Some("C"))
        .expect("the compare form");
    assert_eq!(node.parameter("Q"), Some(1), "one entry");
    assert_eq!(node.parameter("B"), Some(6));
    assert!(node.cost > 0);
    assert!(program.phase1.violations().is_empty());
}

/// The thermometer of a `Packed`'s ℓ is one node, `Ranks(I, valid)`, shared
/// by every consumer that holds the bundle: the fold that scans it and the
/// export that zeroes its tail (`rules/README.md` §4.1, §4.5).
#[test]
fn the_thermometer_of_a_packed_length_is_one_shared_node() {
    let mut block = filtered("x");
    block.push(assign("p", call(Some("np"), "sum", vec![load("t")])));
    // Both values are delivered, so both subgraphs are expanded: the export
    // of the packed filter and the fold that scans it.
    block.push(ret(tuple(vec![load("t"), load("p")])));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    let validity: Vec<_> = requirements(&program, Family::Ranks)
        .into_iter()
        .filter(|(interface, _)| {
            matches!(interface, Interface::Ranks { subject, .. } if subject.valid)
        })
        .collect();
    assert!(!validity.is_empty(), "{}", program.dump());
    for (interface, providers) in &validity {
        assert_eq!(
            providers.len(),
            1,
            "one `PREFIX_RANK/valid` node per bundle: {}",
            interface.render()
        );
    }
    // The node is named by more than one consumer: the export's canonical
    // tail and the fold over the `Packed`.
    let shared = validity.iter().any(|(_, providers)| {
        let node = program.phase1.graph.implementation(providers[0]);
        let output = node.outputs[0];
        program
            .phase1
            .graph
            .implementations
            .iter()
            .filter(|other| other.inputs.contains(&output))
            .count()
            > 1
    });
    assert!(shared, "the thermometer is built once and shared\n{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// Every fixture of this file passes the invariant checker of
/// `revision/phase1/README.md` §2.
#[test]
fn the_invariants_hold_on_every_variable_length_fixture() {
    let programs: Vec<Program> = vec![
        {
            let mut block = filtered("x");
            block.push(ret(call(Some("np"), "sum", vec![load("t")])));
            run(vec![array_input("x", &[8])], block).expect("the filter fixture")
        },
        {
            let mut block = filtered("x");
            block.push(ret(load("t")));
            run(vec![array_input("x", &[8])], block).expect("the packed fixture")
        },
        run(
            vec![array_input("x", &[5])],
            vec![assign("s", call(Some("np"), "sort", vec![load("x")])), ret(load("s"))],
        )
        .expect("the sort fixture"),
    ];
    for program in &programs {
        assert!(
            program.phase1.violations().is_empty(),
            "{:?}\n{}",
            program.phase1.violations(),
            program.dump()
        );
    }
}

/// `np.isclose(a, b, atol=0)` on integer operands is one `eq` per position:
/// the difference is an integer, so the tolerance test is an equality
/// (`map_aligned.md`, the `isclose` row).
#[test]
fn an_integer_isclose_without_a_tolerance_is_one_equality() {
    let program = run(
        vec![array_input("x", &[4]), array_input("y", &[4])],
        vec![ret(call_kw(
            Some("np"),
            "isclose",
            vec![load("x"), load("y")],
            vec![("atol", int(0))],
        ))],
    )
    .expect("the program is accepted");

    let elementwise = nodes(&program, RuleId::MapAligned);
    assert!(!elementwise.is_empty(), "{}", program.dump());
    for node in &elementwise {
        // One `eq` per position, at two units each.
        assert_eq!(node.cost, 4 * 2, "{}", program.dump());
    }
    assert!(program.phase1.violations().is_empty());
}

/// A write whose value is an array is read at the relative coordinates of
/// every slot, under the term `p ∈ R`: a general batch whose entries carry
/// their own live term, which `ENUM_COORDS` answers with one `act(j)` port
/// per entry (`setitem.md`, `enum_coords.md`).
#[test]
fn a_written_region_is_fetched_under_a_per_entry_live_term() {
    let program = run_full(
        vec![array_input("x", &[8]), scalar_input("lo"), scalar_input("hi")],
        vec![
            assign("b", call(Some("np"), "array", vec![load("x")])),
            assign_subscript(
                "b",
                vec![slice_range(Some(load("lo")), Some(load("hi")), None)],
                binop(
                    "mul",
                    subscript(
                        load("b"),
                        vec![slice_range(Some(load("lo")), Some(load("hi")), None)],
                    ),
                    int(2),
                ),
            ),
            ret(call(Some("np"), "sum", vec![load("b")])),
        ],
        vec![requires_between("lo", 0, 8), requires_between("hi", 0, 8)],
    )
    .expect("the program is accepted");

    let writes = nodes(&program, RuleId::SetItem);
    let fetched = writes
        .iter()
        .find(|node| node.variant == Some("fetched"))
        .expect("the written value is an array\n");
    assert_eq!(fetched.parameter("Q"), Some(8));
    // The batch of the `val` port is a general one whose entries carry a
    // live term, so `ENUM_COORDS` raises one `act(j)` port per entry and
    // `TERM_TO_WIRES` compiles the comparisons.
    let general = program
        .phase1
        .graph
        .requirements
        .iter()
        .filter(|node| match &node.id.form {
            Form::Fetch { batch, .. } => matches!(batch, crate::phase1::interface::Batch::General(entries)
                if entries.iter().any(|entry| matches!(entry.live,
                    crate::phase1::interface::Live::Term(_)))),
            _ => false,
        })
        .count();
    assert!(general > 0, "{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// A reshape of a value whose length is decided at run time has a view
/// stride that is not constant: the key form mentions a parameter wire,
/// `π_0 × i_0 + i_1`, which is R4 of `make_ref_runtime.md`, and the reader
/// charges one product per entry.
#[test]
fn a_runtime_view_stride_is_a_parameter_wire() {
    let mut block = filtered("x");
    block.push(assign("w", method(load("t"), "reshape", vec![int(2), int(-1)])));
    block.push(ret(call(Some("np"), "sum", vec![load("w")])));
    let program = run(vec![array_input("x", &[8])], block).expect("the program is accepted");

    let views = nodes(&program, RuleId::MakeRefRuntime);
    assert!(!views.is_empty(), "{}", program.dump());
    let maps: Vec<&Interface> = program
        .phase1
        .graph
        .requirements
        .iter()
        .filter_map(|node| match &node.id.form {
            Form::Represent { interface, .. } if interface.family() == Family::Ref => {
                Some(interface)
            }
            _ => None,
        })
        .collect();
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(reference)
            if reference.map.iter().any(mentions_a_stride_wire))),
        "{:?}",
        maps.iter().map(|interface| interface.render()).collect::<Vec<_>>()
    );
    assert!(program.phase1.violations().is_empty());
}

/// Whether a μ expression multiplies a coordinate by a parameter wire, which
/// is what a run-time stride writes (`rules/README.md` §4.2).
fn mentions_a_stride_wire(expression: &crate::phase1::interface::Mu) -> bool {
    use crate::phase1::interface::Mu;
    match expression {
        Mu::ScaleParam(..) => true,
        Mu::Add(left, right) | Mu::Sub(left, right) => {
            mentions_a_stride_wire(left) || mentions_a_stride_wire(right)
        }
        Mu::Scale(_, inner) | Mu::DivConst(inner, _) | Mu::ModConst(inner, _) => {
            mentions_a_stride_wire(inner)
        }
        _ => false,
    }
}
