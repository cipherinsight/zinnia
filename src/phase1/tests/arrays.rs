//! The fixed-shape array rules of step 2, one fixture per rule and one per
//! engine step the rules forced: the `Fetch` path of
//! `revision/phase1/README.md` §4 step 4, the alignment premise of
//! `rules/README.md` §4.1, the cut of §4 "Acyclicity by construction", and
//! the later-epoch variant of §6.2.

use super::fixtures::*;
use crate::phase1::graph::{ImplNode, Parameter};
use crate::phase1::interface::{Family, Interface, LayoutKind, Mu, MuForm};
use crate::phase1::oracle::questions::Kind;
use crate::phase1::rules::RuleId;

/// Every implementation node of one rule, whatever its variant.
fn nodes<'a>(program: &'a Program, rule: RuleId) -> Vec<&'a ImplNode> {
    program.nodes_of(rule)
}

/// The concrete interfaces a value's row of one family holds.
fn interfaces(program: &Program, family: Family) -> Vec<&Interface> {
    program
        .phase1
        .graph
        .requirements
        .iter()
        .filter_map(|node| match &node.id.form {
            crate::phase1::graph::Form::Represent { interface, .. }
                if interface.family() == family =>
            {
                Some(interface)
            }
            _ => None,
        })
        .collect()
}

/// `a[1:3]` over a fixed-shape input: `MAKE_REF` gives a constant map with no
/// parameter wire, `ENUM_COORDS` proves every entry live, `APPLY_INDEX_MAP`
/// folds every address to a constant and `READ_DIRECT` wires the slots
/// (`make_ref.md`, `read_direct.md`).
#[test]
fn a_constant_slice_is_read_by_wiring() {
    let program = run(
        vec![array_input("a", &[8])],
        vec![ret(subscript(
            load("a"),
            vec![slice_range(Some(int(2)), Some(int(6)), None)],
        ))],
    )
    .expect("the program is accepted");

    // One `Ref` per bundle of the owner, each with `μ_c,0 = 2 + i_0`, no
    // parameter wire, span 1 and cost 0.
    let refs = nodes(&program, RuleId::MakeRef);
    assert!(!refs.is_empty(), "{}", program.dump());
    for node in &refs {
        assert_eq!(node.cost, 0, "a `Ref` allocates no wire");
        assert_eq!(node.parameter("span"), Some(1));
    }
    let maps: Vec<&Interface> = interfaces(&program, Family::Ref);
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if r.params.is_empty()
                && r.map == vec![Mu::Add(Box::new(Mu::Const(2)), Box::new(Mu::Coord(0)))])),
        "{:?}",
        maps
    );

    // The read: every address folds, so `READ_DIRECT` answers at cost 0 and
    // the mux is not offered on that request.
    let direct = nodes(&program, RuleId::ReadDirect);
    assert!(!direct.is_empty(), "{}", program.dump());
    for node in &direct {
        assert_eq!(node.cost, 0);
        assert_eq!(node.parameter("Q"), Some(4));
    }
    assert!(program.phase1.violations().is_empty());
}

/// `BUILD_DENSE` materialises the slice: a box `Store` of the capacity the
/// CONST answers fixed, with the `Packed` attachment of §3.4 over the same
/// wires, and no gate of its own.
#[test]
fn a_dense_store_is_built_from_the_grid_batch() {
    let program = run(
        vec![array_input("a", &[8])],
        vec![ret(subscript(
            load("a"),
            vec![slice_range(Some(int(2)), Some(int(6)), None)],
        ))],
    )
    .expect("the program is accepted");

    let dense = nodes(&program, RuleId::BuildDense);
    assert!(!dense.is_empty(), "{}", program.dump());
    let plain = dense
        .iter()
        .find(|node| node.variant == Some("plain"))
        .expect("the `plain` variant");
    assert_eq!(plain.parameter("Q"), Some(4));
    assert_eq!(plain.parameter("B_0"), Some(4));
    // `plain` allocates nothing: values and flags are the batch's.
    assert_eq!(plain.cost, 0);
    // §3.4: the `Store` and the `Packed` of one node, the box being compact.
    assert!(plain.outputs.len() >= 2, "{}", program.dump());
}

/// A subscript with a run-time start is `MAKE_REF_RUNTIME`: the start is a
/// parameter wire answered by `TERM_TO_WIRES`, and the addresses are wires,
/// so the mux and the ROM answer the read instead of `READ_DIRECT`.
#[test]
fn a_runtime_slice_start_is_a_parameter_wire() {
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

    let runtime = nodes(&program, RuleId::MakeRefRuntime);
    assert!(!runtime.is_empty(), "{}", program.dump());
    assert!(nodes(&program, RuleId::MakeRef).is_empty(), "exactly one of the two");
    let maps = interfaces(&program, Family::Ref);
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if r.params.len() == 1 && !r.is_constant())),
        "the map mentions one parameter wire: {:?}",
        maps
    );
    // A wire address: `READ_DIRECT` does not apply, `READ_KEY_MUX` does.
    assert!(nodes(&program, RuleId::ReadKeyMux).iter().any(|node| node.cost > 0));
    assert!(program.phase1.violations().is_empty());
}

/// `a.T[1:3].reshape(-1)`: the outer view over a `Ref` owner is nested, and
/// `REINDEX_REF` composes the two maps into one over the flat owner
/// (`reindex_ref.md`).
#[test]
fn a_nested_view_is_flattened_into_one_map() {
    let program = run(
        vec![array_input("a", &[4, 5])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![method(
                subscript(
                    method(load("a"), "transpose", vec![]),
                    vec![slice_range(Some(int(1)), Some(int(3)), None)],
                ),
                "reshape",
                vec![int(-1)],
            )],
        ))],
    )
    .expect("the program is accepted");

    let flattened = nodes(&program, RuleId::ReindexRef);
    assert!(!flattened.is_empty(), "{}", program.dump());
    for node in &flattened {
        assert_eq!(node.cost, 0, "a composition allocates no wire");
    }
    // The composite points at a Store-family owner, not at another `Ref`.
    let maps = interfaces(&program, Family::Ref);
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if r.is_flat() && r.span > 1)),
        "{:?}",
        maps
    );
    assert!(program.phase1.violations().is_empty());
}

/// A range batch through a transpose asks CONST once per axis; through a
/// reshape the composed coordinate is one question per entry
/// (`rules/README.md` §3.3, `reading/README.md` §3.3).
///
/// The counts are the questions A holds, which is what §2's invariant makes
/// observable: the transpose composes to a template affine in one grid index
/// per owner axis, the reshape does not.
#[test]
fn a_range_batch_composes_per_axis_and_a_reshape_per_entry() {
    let transposed = run(
        vec![array_input("a", &[4, 5])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![method(load("a"), "transpose", vec![])],
        ))],
    )
    .expect("the program is accepted");
    let reshaped = run(
        vec![array_input("a", &[4, 5])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![method(load("a"), "reshape", vec![int(20)])],
        ))],
    )
    .expect("the program is accepted");

    // Both compose to constant addresses, so both read by wiring.
    for program in [&transposed, &reshaped] {
        assert!(!nodes(program, RuleId::ReadDirect).is_empty(), "{}", program.dump());
        assert!(program.phase1.violations().is_empty());
    }
    // The shape of the composed coordinate is what the two cases differ in:
    // a transpose composes to one template per owner axis, so the map is a
    // permutation of the grid indices; a reshape writes `div` and `mod` out,
    // which is a coordinate of its own per entry.
    let per_axis = interfaces(&transposed, Family::Ref).into_iter().any(|interface| {
        matches!(interface, Interface::Ref(r)
                 if r.map.iter().all(|expression| matches!(expression, Mu::Coord(_))))
    });
    assert!(per_axis, "{:?}", interfaces(&transposed, Family::Ref));
    let per_entry = interfaces(&reshaped, Family::Ref).into_iter().any(|interface| {
        matches!(interface, Interface::Ref(r)
                 if r.map.iter().any(|expression| matches!(
                     expression,
                     Mu::DivConst(..) | Mu::ModConst(..))))
    });
    assert!(per_entry, "{:?}", interfaces(&reshaped, Family::Ref));
}

/// A run-time index: `GATHER` builds the request from the fetched index,
/// `READ_KEY_MUX` and `READ_ROM` both answer it, and the ROM node requires
/// its context, which `ROM_CLOSE_EXACT` provides with `Q_o` as the one
/// ranged parameter of `../README.md` §5.7.
#[test]
fn a_runtime_index_offers_the_mux_and_the_rom_with_its_context() {
    let program = run_full(
        vec![array_input("a", &[16]), array_input("idx", &[4])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![call(Some("np"), "take", vec![load("a"), load("idx")])],
        ))],
        Vec::new(),
    )
    .expect("the program is accepted");

    assert!(!nodes(&program, RuleId::Gather).is_empty(), "{}", program.dump());
    let mux = nodes(&program, RuleId::ReadKeyMux);
    let rom = nodes(&program, RuleId::ReadRom);
    assert!(!mux.is_empty() && !rom.is_empty(), "{}", program.dump());

    // The ROM reader requires the context node, so closure is structural.
    let contexts: Vec<crate::phase1::graph::ReqId> = program
        .phase1
        .graph
        .requirements
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(&node.id.form,
                crate::phase1::graph::Form::Represent { interface, .. }
                    if interface.family() == Family::RomContext)
        })
        .map(|(at, _)| crate::phase1::graph::ReqId(at as u32))
        .collect();
    assert!(!contexts.is_empty(), "the context is a requirement node");
    assert!(
        rom.iter().all(|node| node.inputs.iter().any(|at| contexts.contains(at))),
        "every READ_ROM requires its context"
    );

    // The context's query count is the ranged parameter phase 2 sets.
    let closers = nodes(&program, RuleId::RomCloseExact);
    assert!(!closers.is_empty());
    for node in &closers {
        match node.parameter_of("Q_o") {
            Some(Parameter::Ranged { lo, hi }) => assert!(lo <= hi),
            other => panic!("Q_o is the ranged parameter, got {:?}", other),
        }
    }
    assert!(!nodes(&program, RuleId::RomBind).is_empty());
    assert!(program.phase1.violations().is_empty());
}

/// Two operands of one shape are aligned, so `MAP_ALIGNED` holds the result
/// in their layout with their capacity; the same program read at a batch is
/// `ELEMENTWISE_BATCH`.
#[test]
fn alignment_proved_gives_a_slotwise_node() {
    let program = run(
        vec![array_input("a", &[4]), array_input("b", &[4])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![binop("add", load("a"), load("b"))],
        ))],
    )
    .expect("the program is accepted");

    let aligned = nodes(&program, RuleId::MapAligned);
    assert!(!aligned.is_empty(), "{}", program.dump());
    for node in &aligned {
        assert_eq!(node.parameter("B"), Some(4));
        // `add` on I64 is one `wrap_64` with a two-bit carry per slot.
        assert_eq!(node.cost, 4 * 66);
        assert_eq!(node.inputs.len(), 2, "one port per array operand");
    }
    assert!(!nodes(&program, RuleId::ElementwiseBatch).is_empty());
    assert!(program.phase1.violations().is_empty());
}

/// Two operands whose extents are not provably equal are not aligned, so
/// `MAP_ALIGNED` offers nothing and the value is represented by
/// `BUILD_DENSE` over `ELEMENTWISE_BATCH` (`map_aligned.md`, premises).
#[test]
fn alignment_refuted_leaves_the_batch_route() {
    let program = run(
        vec![array_input("a", &[4, 1]), array_input("b", &[1, 3])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![binop("mul", load("a"), load("b"))],
        ))],
    )
    .expect("the program is accepted");

    assert!(
        nodes(&program, RuleId::MapAligned).is_empty(),
        "the operands broadcast, so no slot holds the same position in both"
    );
    assert!(!nodes(&program, RuleId::BroadcastMap).is_empty(), "{}", program.dump());
    assert!(!nodes(&program, RuleId::ElementwiseBatch).is_empty());
    assert!(program.phase1.violations().is_empty());
}

/// `BROADCAST_MAP` builds the dense box of the broadcast result: one gadget
/// per slot of the capacity box, the operands read at the projected
/// coordinates.
#[test]
fn broadcasting_between_arrays_builds_a_box() {
    let program = run(
        vec![array_input("a", &[4, 1]), array_input("b", &[1, 3])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![binop("mul", load("a"), load("b"))],
        ))],
    )
    .expect("the program is accepted");

    let broadcast = nodes(&program, RuleId::BroadcastMap);
    assert!(!broadcast.is_empty(), "{}", program.dump());
    for node in &broadcast {
        assert_eq!(node.parameter("B"), Some(12), "the box is 4 × 3");
        // `mul` on I64: one `mul` and one `wrap_64` with a 64-bit carry.
        assert_eq!(node.cost, 12 * (1 + 128));
    }
    assert!(program.phase1.violations().is_empty());
}

/// A reduction: `EXPAND_FIBERS` lays the operand out once per provider and
/// `FOLD_GROUPS` folds each, so one requirement holds one fold node per
/// layout (`reduction/README.md`).
#[test]
fn a_reduction_offers_one_fold_per_layout() {
    let program = run(
        vec![array_input("a", &[4])],
        vec![ret(call(Some("np"), "sum", vec![load("a")]))],
    )
    .expect("the program is accepted");

    let fibers = nodes(&program, RuleId::ExpandFibers);
    let folds = nodes(&program, RuleId::FoldGroups);
    assert!(fibers.len() >= 3, "box, ranked and compact: {}", program.dump());
    // One fold per distinct `Groups` requirement: two layouts that give the
    // same interface are one requirement with two providers, and the fold
    // above them is one node.
    let layouts = interfaces(&program, Family::Groups).len();
    assert_eq!(folds.len(), layouts, "one fold per layout: {}", program.dump());
    for node in &folds {
        assert_eq!(node.parameter("P"), Some(1), "a scalar reduction has one fiber");
        assert_eq!(node.parameter("W"), Some(4));
        assert_eq!(node.inputs.len(), 1, "the `groups` port");
    }
    // The four folds answer one requirement: the sum's wire.
    let outputs: Vec<_> = folds.iter().map(|node| node.outputs.clone()).collect();
    assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]), "{:?}", outputs);
    assert!(program.phase1.violations().is_empty());
}

/// A reduction over one axis of a rank-2 value: P fibers of W slots, with the
/// output a box `Store` of P slots.
#[test]
fn a_partial_reduction_keeps_the_outer_axes() {
    let program = run(
        vec![array_input("a", &[2, 3])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![call_kw(Some("np"), "sum", vec![load("a")], vec![("axis", int(1))])],
        ))],
    )
    .expect("the program is accepted");

    let folds = nodes(&program, RuleId::FoldGroups);
    let partial = folds
        .iter()
        .find(|node| node.parameter("P") == Some(2))
        .expect("the fold over axis 1 has two fibers");
    assert_eq!(partial.parameter("W"), Some(3));
    assert!(program.phase1.violations().is_empty());
}

/// A contraction: one product per (output position, k) and one wrap per
/// output position (`matmul.md`, emitter).
#[test]
fn a_contraction_folds_k_products_per_position() {
    let program = run(
        vec![array_input("a", &[2, 3]), array_input("b", &[3, 4])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![binop("mat_mul", load("a"), load("b"))],
        ))],
    )
    .expect("the program is accepted");

    let product = nodes(&program, RuleId::Matmul);
    assert_eq!(product.len(), 1, "{}", program.dump());
    assert_eq!(product[0].parameter("K"), Some(3));
    assert_eq!(product[0].parameter("Q"), Some(8), "2 × 4 output positions");
    assert!(product[0].cost > 0);
    assert!(program.phase1.violations().is_empty());
}

/// A scan produces its result in the interface of the slots it read
/// (`cumsum.md`, `produced_output_bundle`).
#[test]
fn a_scan_keeps_the_slots_of_its_operand() {
    let program = run(
        vec![array_input("a", &[4])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![call(Some("np"), "cumsum", vec![load("a")])],
        ))],
    )
    .expect("the program is accepted");

    let scans = nodes(&program, RuleId::Cumsum);
    assert!(!scans.is_empty(), "{}", program.dump());
    for node in &scans {
        assert_eq!(node.parameter("W"), Some(4));
        assert!(node.cost > 0, "one wrap per slot");
    }
    assert!(program.phase1.violations().is_empty());
}

/// A value-sized constant: the capacity is `MAX ℓ`, which the `var_range`
/// of the declaration bounds by the declared maximum, and the validity is a
/// thermometer over it (`constant_sized.md`).
#[test]
fn a_value_sized_constant_takes_its_capacity_from_max() {
    let program = run_full(
        vec![scalar_input("k")],
        vec![
            assign("z", call(Some("np"), "arange", vec![load("k")])),
            ret(call(None, "len", vec![load("z")])),
        ],
        vec![requires_between("k", 0, 64)],
    )
    .expect("the program is accepted");

    let sized = nodes(&program, RuleId::ConstantSized);
    assert!(!sized.is_empty(), "{}", program.dump());
    for node in &sized {
        assert_eq!(node.parameter("B"), Some(64), "the declared maximum");
        // `therm(ℓ over 64)` is `2·64 − 1`.
        assert_eq!(node.cost, 127);
    }
    assert!(program.phase1.stats.kind(Kind::Max).asked > 0, "the capacity is a MAX");
    assert!(program.phase1.violations().is_empty());
}

/// The later-epoch variant of §6.2: a check that tightens the length gives a
/// second node whose capacity is the MAX over the larger Φ_e, and the default
/// node stays beside it. Without such a check only one node is offered.
#[test]
fn a_later_epoch_variant_appears_only_when_the_max_is_smaller() {
    let tightened = run_full(
        vec![scalar_input("k")],
        vec![
            assign("z", call(Some("np"), "arange", vec![load("k")])),
            // A check over the *length* is a term of Φ over ℓ, so Φ_{e′}
            // bounds the capacity; a check over the data scalar `k` would be
            // an opaque Boolean and bound nothing.
            assert_stmt(binop("lte", call(None, "len", vec![load("z")]), int(8))),
            ret(call(None, "len", vec![load("z")])),
        ],
        vec![requires_between("k", 0, 64)],
    )
    .expect("the program is accepted");
    let plain = run_full(
        vec![scalar_input("k")],
        vec![
            assign("z", call(Some("np"), "arange", vec![load("k")])),
            ret(call(None, "len", vec![load("z")])),
        ],
        vec![requires_between("k", 0, 64)],
    )
    .expect("the program is accepted");

    let capacities = |program: &Program| {
        let mut out: Vec<i128> = nodes(program, RuleId::ConstantSized)
            .iter()
            .filter_map(|node| node.parameter("B"))
            .collect();
        out.sort();
        out.dedup();
        out
    };
    assert_eq!(capacities(&plain), vec![64], "no check, so one capacity");
    assert_eq!(
        capacities(&tightened),
        vec![8, 64],
        "the check tightens the MAX, so the variant is offered beside the default: {}",
        tightened.dump()
    );
    // The variant assumed the check, so its epoch is the check's position.
    let variant = nodes(&tightened, RuleId::ConstantSized)
        .into_iter()
        .find(|node| node.parameter("B") == Some(8))
        .expect("the tight variant");
    assert!(variant.epoch >= 1, "the variant assumed the check");
    assert!(tightened.phase1.violations().is_empty());
}

/// A scalar index that is not constant is normalised once, by
/// `INDEX_NORMALIZE`, whose `ok` bit is the wire the check reads
/// (`index_normalize.md`, `term_to_wires.md` P3).
#[test]
fn a_runtime_index_is_normalised_once() {
    let program = run(
        vec![array_input("a", &[8]), scalar_input("i")],
        vec![ret(subscript(load("a"), vec![load("i")]))],
    )
    .expect("the program is accepted");

    let normalised = nodes(&program, RuleId::IndexNormalize);
    assert_eq!(normalised.len(), 1, "one node for the pair ⟨i, d⟩: {}", program.dump());
    // `/full`: the sign test, the two bound comparisons and their `and`.
    assert_eq!(normalised[0].variant, Some("/full"));
    assert!(normalised[0].outputs.len() >= 2, "the pair and the `ok` bit");
    assert!(program.phase1.violations().is_empty());
}

/// Reading a value to rebuild it is the cycle of §4: `BUILD_DENSE` over a
/// `Fetch` of v whose only provider would be that `BUILD_DENSE`. The cut
/// drops that candidate, the row is provisional, and a later consumer outside
/// the cycle fills it again unrestricted.
#[test]
fn reading_a_value_to_rebuild_it_is_cut() {
    let program = run(
        vec![array_input("a", &[4])],
        vec![
            assign("b", binop("add", load("a"), load("a"))),
            ret(binop("add", load("b"), load("b"))),
        ],
    )
    .expect("the program is accepted");

    // G stays acyclic, which is what the cut is for.
    assert!(program.phase1.violations().is_empty(), "{}", program.dump());
    // The row that dropped the candidate is provisional, so it is not
    // memoised and its requirement nodes are its own; a later consumer
    // outside the cycle fills it again with every provider.
    let provisional = program
        .phase1
        .graph
        .requirements
        .iter()
        .filter(|node| node.id.provisional.is_some())
        .count();
    assert!(provisional > 0, "the cut made a row provisional: {}", program.dump());
    // Both routes exist: recomputing the value at the batch, and reading a
    // bundle of it.
    assert!(!nodes(&program, RuleId::ElementwiseBatch).is_empty(), "{}", program.dump());
    assert!(
        !nodes(&program, RuleId::ReadDirect).is_empty(),
        "the unrestricted fill offers the read: {}",
        program.dump()
    );
    // No node of G reads a requirement it provides.
    for node in program.phase1.graph.live_implementations() {
        let node = program.phase1.graph.implementation(node);
        for output in &node.outputs {
            assert!(
                !node.inputs.contains(output),
                "{} reads the requirement it answers",
                node.rule.as_str()
            );
        }
    }
}

/// The three read mechanisms over one owner: a dense owner with constant
/// addresses is `READ_DIRECT`, a ranked one is the mux or the ROM, and the
/// `ReadReq` of each provider is its own requirement (`reading/README.md`
/// §2).
#[test]
fn one_read_request_per_provider_of_a_value() {
    let program = run(
        vec![array_input("a", &[4])],
        vec![ret(call(Some("np"), "sum", vec![load("a")]))],
    )
    .expect("the program is accepted");

    let requests = interfaces(&program, Family::ReadReq);
    assert!(requests.len() >= 2, "one per bundle of a: {:?}", requests);
    let dense = requests.iter().filter(|interface| {
        matches!(interface, Interface::ReadReq { owner, constant, .. }
                 if owner.is_dense() && *constant)
    });
    assert!(dense.count() >= 1);
    assert!(!nodes(&program, RuleId::ReadDirect).is_empty());
    assert!(!nodes(&program, RuleId::ReadKeyMux).is_empty(), "the ranked owner");
    assert!(program.phase1.violations().is_empty());
}

/// `ENUM_COORDS` is one node per (value, batch), shared by every owner and
/// every read mechanism of that `Fetch` (`enum_coords.md`,
/// `effect_dependencies`).
#[test]
fn the_coordinates_of_a_batch_are_one_node() {
    let program = run(
        vec![array_input("a", &[4])],
        vec![ret(call(Some("np"), "sum", vec![load("a")]))],
    )
    .expect("the program is accepted");

    let coordinates = nodes(&program, RuleId::EnumCoords);
    assert_eq!(coordinates.len(), 1, "{}", program.dump());
    // Every entry of the grid is live and `g` is `true`, so the flags are the
    // constant 1 and the node costs nothing.
    assert_eq!(coordinates[0].cost, 0);
    assert_eq!(coordinates[0].parameter("Q"), Some(4));
    let maps = nodes(&program, RuleId::ApplyIndexMap);
    assert!(maps.len() >= 2, "one per provider of a");
    for node in &maps {
        assert!(node.inputs.contains(&coordinates[0].outputs[0]));
    }
}

/// A product of two run-time extents is `SHAPE_CHECK`'s wire: the term
/// language has no product of two variables, so phase 0 introduced σ and the
/// rule multiplies the factors (`shape_check.md`).
#[test]
fn a_product_of_runtime_extents_is_one_multiplication() {
    let program = run(
        vec![varlen_input_2d("x", &[("N", 8), ("M", 8)])],
        vec![ret(method(load("x"), "size", vec![]))],
    )
    .expect("the program is accepted");

    let products = nodes(&program, RuleId::ShapeCheck);
    assert_eq!(products.len(), 1, "one node for σ: {}", program.dump());
    assert_eq!(products[0].inputs.len(), 2, "one port per run-time factor");
    // `z_{j+1} = mul(z_j, f_{j+1})`: m − 1 products for m factors.
    assert_eq!(products[0].cost, 1);
    // The width is the sum of the factors' widths, below `Ω.max_width`.
    assert_eq!(products[0].parameter("w"), Some(8));
    assert!(program.phase1.violations().is_empty());
}

/// `a[None, :]` and `transpose(a, (1, 0))`: a `None` entry adds an output
/// axis of extent 1 over no owner axis, and an explicit permutation is read
/// off the call (`make_ref.md`, `source_pattern`).
#[test]
fn a_new_axis_and_an_explicit_permutation_are_constant_maps() {
    let inserted = run(
        vec![array_input("a", &[4])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![subscript(load("a"), vec![new_axis(), slice_range(None, None, None)])],
        ))],
    )
    .expect("the program is accepted");
    let permuted = run(
        vec![array_input("a", &[2, 3])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![call(Some("np"), "transpose", vec![load("a"), ints(&[1, 0])])],
        ))],
    )
    .expect("the program is accepted");

    for program in [&inserted, &permuted] {
        assert!(!nodes(program, RuleId::MakeRef).is_empty(), "{}", program.dump());
        assert!(!nodes(program, RuleId::ReadDirect).is_empty(), "every address folds");
        assert!(program.phase1.violations().is_empty());
    }
    // The `None` entry contributes no owner coordinate: the map is the one
    // coordinate of the owner's single axis.
    let maps = interfaces(&inserted, Family::Ref);
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if r.map == vec![Mu::Coord(1)])),
        "{:?}",
        maps
    );
}

/// The invariant checker of §2 passes on every fixture of this module.
#[test]
fn the_invariants_hold_on_every_fixture() {
    let programs = vec![
        run(
            vec![array_input("a", &[8])],
            vec![ret(subscript(
                load("a"),
                vec![slice_range(Some(int(2)), Some(int(6)), None)],
            ))],
        ),
        run(
            vec![array_input("a", &[4]), array_input("b", &[4])],
            vec![ret(binop("add", load("a"), load("b")))],
        ),
        run(
            vec![array_input("a", &[2, 3])],
            vec![ret(call(Some("np"), "sum", vec![method(load("a"), "transpose", vec![])]))],
        ),
        run(
            vec![array_input("a", &[16]), array_input("idx", &[4])],
            vec![ret(call(
                Some("np"),
                "take",
                vec![load("a"), load("idx")],
            ))],
        ),
        run(
            vec![array_input("a", &[2, 3]), array_input("b", &[3, 4])],
            vec![ret(binop("mat_mul", load("a"), load("b")))],
        ),
        run(
            vec![array_input("a", &[4])],
            vec![ret(call(Some("np"), "cumsum", vec![load("a")]))],
        ),
        run(
            vec![array_input("a", &[4, 1]), array_input("b", &[1, 3])],
            vec![ret(binop("mul", load("a"), load("b")))],
        ),
    ];
    for (at, outcome) in programs.into_iter().enumerate() {
        let program = outcome.unwrap_or_else(|why| panic!("fixture {} rejected: {}", at, why));
        assert!(
            program.phase1.violations().is_empty(),
            "fixture {}: {:?}\n{}",
            at,
            program.phase1.violations(),
            program.dump()
        );
        // Every parameter of every node is an integer or the one range
        // `../README.md` §5.7 admits, which is the typed form of "no node
        // mentions a variable of Φ".
        for node in &program.phase1.graph.implementations {
            for (name, value) in &node.parameters {
                if let Parameter::Ranged { lo, hi } = value {
                    assert!(lo <= hi, "{} has an empty range", name);
                }
            }
        }
    }
}

/// A delivered array is exported from a bundle whose keys are constants: the
/// box `Store` or the `Packed`, never a `Ref` and never a ranked layout
/// (`export.md`).
#[test]
fn a_delivered_array_is_exported_from_a_dense_bundle() {
    let program = run(
        vec![array_input("a", &[4]), array_input("b", &[4])],
        vec![ret(binop("add", load("a"), load("b")))],
    )
    .expect("the program is accepted");

    let exports = nodes(&program, RuleId::Export);
    assert!(!exports.is_empty(), "{}", program.dump());
    for node in &exports {
        assert_eq!(node.parameter("B"), Some(4));
        assert_eq!(node.parameter("cells"), Some(4), "no run-time extent to output");
    }
    // The bundles the export reads are dense.
    for node in &exports {
        for input in &node.inputs {
            let form = &program.phase1.graph.node(*input).id.form;
            if let crate::phase1::graph::Form::Represent { interface, .. } = form {
                if let Some(lambda) = interface.lambda() {
                    assert!(
                        matches!(lambda.layout, LayoutKind::Box(_) | LayoutKind::Compact(_)),
                        "{:?}",
                        lambda.layout
                    );
                }
            }
        }
    }
}

/// A reshape over a dense owner is the key form of §2.3 when the owner's key
/// is the C-order index, and the coordinate form over a box owner
/// (`make_ref.md`, `produced_output_bundle`).
#[test]
fn a_reshape_writes_the_key_form_over_a_compact_owner() {
    let program = run(
        vec![array_input("a", &[4, 5])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![method(load("a"), "reshape", vec![int(20)])],
        ))],
    )
    .expect("the program is accepted");

    let maps = interfaces(&program, Family::Ref);
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if matches!(r.form, MuForm::Key))),
        "the compact owner takes the key form: {:?}",
        maps
    );
    assert!(
        maps.iter().any(|interface| matches!(interface, Interface::Ref(r)
            if matches!(r.form, MuForm::Coordinate { .. }))),
        "the box owner takes the coordinate form: {:?}",
        maps
    );
    assert!(program.phase1.violations().is_empty());
}


// ── The term arena (`revision/phase1/README.md` §1.1) ────────────────

/// CONST on a composed coordinate, through a slice of a transpose.
///
/// `a.T[1:3]` reads `a` at `(j, i + 1)` for the output coordinate `(i, j)`.
/// Phase 1 builds that composition as a term and asks CONST on it — one
/// question per owner axis for a range batch, which the affine map this is
/// makes enough (`reading/README.md`, "CONST on a composed coordinate").
/// Every address folds, so `READ_DIRECT` answers the `Fetch` and the read
/// costs nothing.
#[test]
fn a_composed_coordinate_through_a_view_is_constant_per_axis() {
    let program = run(
        vec![array_input("a", &[4, 5])],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![subscript(
                method(load("a"), "transpose", vec![]),
                vec![slice_range(Some(int(1)), Some(int(3)), None)],
            )],
        ))],
    )
    .expect("the program is accepted");

    // The read request folded: `constant` is set on the `ReadReq` interface
    // and `READ_DIRECT` is the mechanism.
    let requests = interfaces(&program, Family::ReadReq);
    assert!(
        requests
            .iter()
            .any(|interface| matches!(interface, Interface::ReadReq { constant: true, .. })),
        "every composed address is a compile-time constant: {:?}\n{}",
        requests,
        program.dump()
    );
    assert!(!nodes(&program, RuleId::ReadDirect).is_empty(), "{}", program.dump());
    assert!(
        nodes(&program, RuleId::ReadKeyMux).is_empty(),
        "a constant address needs no mux"
    );

    // Every CONST the composition asked was answered without the solver:
    // the coordinates are literals and the arena folds the composition, so
    // the question never leaves the syntactic layer.
    for answer in program.answers_of(Kind::Const) {
        assert_ne!(
            answer.layer,
            crate::phase1::oracle::answers::Layer::Solver,
            "a composed coordinate over literals is decided without Z3"
        );
    }
    assert!(program.phase1.violations().is_empty());
}

/// Alignment is proved on a built equality.
///
/// Two symbolic arrays that share a shape symbol have one hash-consed
/// dimension term, which needs no question; two that do not are compared by
/// `PROVE(dims_a[0] = dims_b[0])` on the term phase 1 builds
/// (`rules/README.md` §6.3). Here the `@requires` makes the two lengths
/// equal, so the PROVE succeeds and `MAP_ALIGNED` offers a node.
#[test]
fn alignment_is_proved_on_a_built_equality() {
    let program = run_full(
        vec![varlen_input("a", "N", 8), varlen_input("b", "M", 8)],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![binop("add", load("a"), load("b"))],
        ))],
        vec![requires_le("N", "M"), requires_le("M", "N")],
    )
    .expect("the program is accepted");

    // The equality is a term phase 1 built, and the oracle proved it.
    let proofs: Vec<_> = program
        .answers_of(Kind::Prove)
        .into_iter()
        .filter(|answer| {
            answer.key.question.nodes().iter().any(|&at| {
                at as usize >= program.phase1.base_nodes
                    && matches!(
                        program.phase1.terms.node_at(at as usize),
                        crate::phase0::term::NodeView::Eq(..)
                    )
            })
        })
        .collect();
    assert!(
        !proofs.is_empty(),
        "the alignment premise is a built equality\n{}",
        program.dump()
    );
    assert!(
        proofs
            .iter()
            .any(|answer| answer.verdict == crate::phase1::oracle::answers::Verdict::Proved),
        "the `@requires` pair makes the two extents equal: {:?}",
        proofs.iter().map(|answer| &answer.verdict).collect::<Vec<_>>()
    );
    assert!(!nodes(&program, RuleId::MapAligned).is_empty(), "{}", program.dump());
    assert!(program.phase1.violations().is_empty());
}

/// A strided range batch carries a thermometer.
///
/// `a[1::2]` over an array of run-time length is the template `1 + 2·j`,
/// which step 2 refused because only the grid template `j ↦ j` had a length
/// that was a term of Φ. Phase 1 now builds the length through phase 0's
/// slice-length helper, `max(0, (dims_a[0] − 1 + 2 − 1) div 2)`, folds `g`
/// into it and emits one `therm` over the axis (`enum_coords.md`).
#[test]
fn a_strided_batch_is_admitted_with_a_thermometer() {
    let program = run(
        vec![varlen_input("a", "N", 8)],
        vec![ret(call(
            Some("np"),
            "sum",
            vec![subscript(
                load("a"),
                vec![slice_range(Some(int(1)), None, Some(int(2)))],
            )],
        ))],
    )
    .expect("the program is accepted");

    let coords = nodes(&program, RuleId::EnumCoords);
    assert!(!coords.is_empty(), "{}", program.dump());
    assert!(
        coords.iter().any(|node| node.cost > 0),
        "the thermometer of the strided axis is emitted: {:?}\n{}",
        coords.iter().map(|node| node.cost).collect::<Vec<_>>(),
        program.dump()
    );
    assert!(program.phase1.stats.terms_built > 0, "the length is a built term");
    assert!(program.phase1.violations().is_empty());
}

/// `a.shape[k]` is a dimension term whose value names no array.
///
/// Phase 0 records it as an `E-Property` tuple and an `E-Composite-Index`
/// into it, so the operand of the value is the tuple. `LEN_SHAPE` resolves
/// the term instead — the variable it is names the array (`rules/README.md`
/// §2) — and the extent wire of that array's bundle answers the row. Without
/// it a check whose defcond mentions the symbol has no wire, which is how
/// the two `npbench` kernels of step 2's sweep were rejected.
#[test]
fn a_shape_scalar_through_a_composite_index_has_a_wire() {
    let program = run_full(
        vec![varlen_input("a", "N", 8), scalar_input("k")],
        vec![
            assign("n", subscript(method(load("a"), "shape", vec![]), vec![int(0)])),
            ret(binop("add", load("n"), load("k"))),
        ],
        vec![requires_between("k", 0, 3)],
    )
    .expect("the program is accepted");

    let shapes = nodes(&program, RuleId::LenShape);
    assert!(!shapes.is_empty(), "LEN_SHAPE answers the shape scalar\n{}", program.dump());
    assert!(
        shapes.iter().any(|node| node.variant == Some("/read")),
        "the wire is the input bundle's extent wire: {:?}\n{}",
        shapes.iter().map(|node| node.variant).collect::<Vec<_>>(),
        program.dump()
    );
    assert!(program.phase1.violations().is_empty());
}
