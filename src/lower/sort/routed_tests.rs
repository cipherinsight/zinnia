//! The routed Waksman sort (`routed`): NumPy's stable sort on edge shapes,
//! every setting of the switches either rejected or giving NumPy's outputs
//! (the switch bits' freedom changes no output), non-bit settings rejected,
//! and the cost rule choosing it only where it is cheaper. The halo2-base
//! versions of the tamper tests are in `prove/halo2_base/synthesizer_tests.rs`
//! (`routing_settings_are_rejected_or_give_numpys_outputs`); the plan is not
//! offered on Noir.

use super::cost::{routed_plan, unit_costs, Demand};
use super::test_support::{
    hints_as_inputs, inputs_with_hints, mock_run, sort_graph, stable_sort, Forced, Output, PAYLOAD_BASE,
};
use super::waksman;
use crate::ir_defs::{HintKind, IR};
use crate::lower::Target;

fn every_output(n: u32) -> Vec<Output> {
    (0..n).flat_map(|r| [Output::Value(r), Output::Index(r)]).collect()
}

/// Every payload entry, value and index of every rank (the payload first:
/// `sort_graph` reads the payload inputs at its first payload output, and a
/// sort's operands must precede its lanes).
fn values_indices_payloads(n: u32) -> Vec<Output> {
    (0..n).map(Output::Payload).chain(every_output(n)).collect()
}

/// The routed plan on `values` gives NumPy's outputs on the mock (no
/// proof where an output does not exist: a runtime rank out of range).
fn assert_routed_is_numpy(values: &[i64], outputs: &[Output]) {
    let sort = sort_graph(values, outputs, Forced::Routed);
    assert!(!sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))), "a placeholder survived");
    let (ok, got) = mock_run(&sort.graph, &sort.inputs);
    match &sort.expected {
        Some(expected) => {
            assert!(ok, "{values:?} {outputs:?}: honest run rejected");
            assert_eq!(&got, expected, "{values:?} {outputs:?}");
        }
        None => assert!(!ok, "{values:?} {outputs:?}: an out-of-range rank was accepted"),
    }
}

/// Edge shapes: one element, two, sizes that are not powers of two (the
/// network's odd and even recursions), all elements equal, already sorted,
/// reversed, ties, and the ends of the signed 64-bit window.
fn edge_inputs() -> Vec<Vec<i64>> {
    let mut inputs = vec![
        vec![4],
        vec![5, 5],
        vec![2, 1],
        vec![1, 2],
        vec![3, 1, 2],
        vec![7; 7],
        (0..9).collect(),
        (0..9).rev().collect(),
        vec![5, -3, 5, 0, -3, 7, 5, 0, -9],
        vec![i64::MAX, i64::MIN, 0, i64::MAX, -1, i64::MIN],
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for n in [5usize, 6, 11, 16, 33, 100] {
        inputs.push(
            (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) % 7) as i64 - 3 // many ties
                })
                .collect(),
        );
    }
    inputs
}

#[test]
fn routed_sort_equals_numpy_on_edge_shapes() {
    for values in edge_inputs() {
        let n = values.len() as u32;
        assert_routed_is_numpy(&values, &every_output(n));
        assert_routed_is_numpy(&values, &(0..n).map(Output::Index).collect::<Vec<_>>());
        assert_routed_is_numpy(&values, &(0..n).map(Output::Value).collect::<Vec<_>>());
        assert_routed_is_numpy(&values, &(0..n).map(Output::Payload).collect::<Vec<_>>());
        // A few outputs only, and runtime ranks (in range and not).
        assert_routed_is_numpy(&values, &[Output::Payload(0), Output::Index(n - 1), Output::Value(n / 2)]);
        for r in [0, i64::from(n) - 1, -1, i64::from(n)] {
            assert_routed_is_numpy(&values, &[Output::ValueAt(r), Output::IndexAt(r)]);
        }
    }
}

/// The routed graph has exactly one hint per switch, each a routing bit.
#[test]
fn every_switch_is_one_routing_bit() {
    for n in [1usize, 2, 3, 4, 5, 9, 100] {
        let values: Vec<i64> = (0..n as i64).rev().collect();
        let sort = sort_graph(&values, &every_output(n as u32), Forced::Routed);
        let hints: Vec<&IR> = sort.graph.stmts.iter().map(|s| &s.ir).filter(|ir| matches!(ir, IR::Hint { .. })).collect();
        assert_eq!(hints.len(), waksman::switch_count(n), "n = {n}");
        assert!(hints.iter().all(|ir| matches!(ir, IR::Hint { kind: HintKind::RoutingBits, .. })));
        let bit_checks = sort.graph.stmts.iter().filter(|s| matches!(s.ir, IR::AssertBit)).count();
        assert_eq!(bit_checks, hints.len(), "n = {n}: every switch bit asserted a bit");
    }
}

/// Whether the switch setting `setting` (bit `t` = switch `t` crossed)
/// routes the keys `(a_j, j)` into ascending order.
fn setting_sorts(values: &[i64], setting: u32) -> bool {
    let wires: Vec<(i64, usize)> = values.iter().copied().zip(0..).collect();
    let mut t = 0;
    let routed = waksman::apply(wires, &mut |x, y| {
        let crossed = setting >> t & 1 == 1;
        t += 1;
        if crossed { (y, x) } else { (x, y) }
    });
    routed.windows(2).all(|w| w[0] < w[1])
}

/// Every setting of the switches of small networks, played by a prover that
/// chooses the bits freely: a setting is accepted exactly when it routes the
/// keys into ascending order, and then the outputs are NumPy's. So every
/// wrong order of the keys (the network realizes every permutation, so every
/// order is among the settings), an unstable order of ties included, is
/// rejected, and the settings that are accepted are interchangeable. Some
/// input must have two accepted settings, or the test would not exercise the
/// switch bits' freedom.
#[test]
fn every_routing_setting_is_rejected_or_gives_numpys_outputs() {
    let inputs: Vec<Vec<i64>> = vec![
        vec![7, 7],
        vec![2, 1, 2],
        vec![7, 7, 7, 7],
        vec![0, 1, 2, 3],
        vec![3, 2, 1, 0],
        vec![3, 1, 3, 0],
        vec![2, 2, 1, 2, 1],
        vec![4, 0, 4, 0, 4],
        vec![1, 1, 1, 1, 1, 1],
        vec![5, 4, 3, 2, 1, 0],
        vec![0, 2, 0, 2, 1, 1],
    ];
    let mut free_inputs = 0;
    for values in inputs {
        let n = values.len();
        let sort = sort_graph(&values, &values_indices_payloads(n as u32), Forced::Routed);
        let expected = sort.expected.clone().expect("every output exists");
        let (graph, sites) = hints_as_inputs(&sort);
        let switches = waksman::switch_count(n);
        assert_eq!(sites.len(), switches);
        let mut accepted = 0;
        for setting in 0u32..(1 << switches) {
            let bits: Vec<i64> = (0..switches).map(|t| i64::from(setting >> t & 1)).collect();
            let (ok, got) = mock_run(&graph, &inputs_with_hints(&sort, &sites, &bits));
            assert_eq!(ok, setting_sorts(&values, setting), "{values:?}: setting {setting:b}");
            if ok {
                accepted += 1;
                assert_eq!(got, expected, "{values:?}: setting {setting:b} accepted with other outputs");
            }
        }
        assert!(accepted >= 1, "{values:?}: no setting accepted");
        if accepted > 1 {
            free_inputs += 1;
        }
    }
    assert!(free_inputs > 0, "no input has two accepted settings: the freedom was not exercised");
}

/// A switch whose bit is not 0 or 1 would mix its two wires (a bit of 1/2
/// makes both outputs `(x + y) / 2`, a duplicate): every such value, with
/// the other switches honest, is rejected.
#[test]
fn non_bit_switch_settings_are_rejected() {
    let values = [3, 1, 3, 0, 1];
    let sort = sort_graph(&values, &values_indices_payloads(5), Forced::Routed);
    let (graph, sites) = hints_as_inputs(&sort);
    let honest: Vec<i64> = sites.iter().map(|s| s.honest).collect();
    assert!(mock_run(&graph, &inputs_with_hints(&sort, &sites, &honest)).0, "honest setting rejected");
    for t in 0..sites.len() {
        for wrong in [2, -1, 3, 1 << 32, i64::MIN] {
            let mut bits = honest.clone();
            bits[t] = wrong;
            let (ok, got) = mock_run(&graph, &inputs_with_hints(&sort, &sites, &bits));
            assert!(!ok, "switch {t} = {wrong} accepted, outputs {got:?}");
        }
    }
}

/// The sorted payload entries are those of NumPy's stable order even when
/// every key is equal (a payload distinguishes the elements the values do
/// not).
#[test]
fn equal_keys_carry_payloads_in_original_order() {
    let values = [9i64; 6];
    let sort = sort_graph(&values, &(0..6).map(Output::Payload).collect::<Vec<_>>(), Forced::Routed);
    let (ok, got) = mock_run(&sort.graph, &sort.inputs);
    assert!(ok);
    assert_eq!(got, (0..6).map(|j| PAYLOAD_BASE + j).collect::<Vec<_>>());
    assert_eq!(stable_sort(&values).1, (0..6).collect::<Vec<_>>());
}

fn full_demand(n: usize) -> Demand {
    Demand { n, indices: (0..n).collect(), ..Demand::default() }
}

#[test]
fn routed_plan_is_offered_on_halo2_base_only_and_from_two_elements() {
    for n in [1usize, 2, 3, 100, 1024] {
        assert!(routed_plan(&full_demand(n), &unit_costs(Target::Noir)).is_none(), "Noir, n = {n}");
        assert_eq!(routed_plan(&full_demand(n), &unit_costs(Target::Halo2Base)).is_some(), n >= 2, "n = {n}");
    }
}

/// The cost rule against measurement: for argsorts, sorts, both and payload
/// gathers of 2 to 64 elements, the plan the halo2-base rule picks measures
/// within 5% of the cheapest of the plans it weighs (Batcher's networks,
/// the adjacent-swap network, rank selectors, the permutation check, the
/// routed network), and the routed plan is picked only where it measures
/// cheaper than every other. Cells are advice + lookup at lookup width 8,
/// plus a table group's fixed columns when the reads at hinted ranks became
/// table reads (the price the layer weighs them at, `memory::cost::table`;
/// advice cells alone do not see a table's columns).
#[cfg(feature = "halo2-base-backend")]
#[test]
fn halo2_base_cost_rule_picks_a_measured_cheapest_plan() {
    let cells = |values: &[i64], outputs: &[Output], forced: Forced| {
        let sort = sort_graph(values, outputs, forced);
        let stats = crate::prove::halo2_base::cell_stats(&sort.graph, 8, 32).expect("cells");
        let tables = sort.graph.stmts.iter().any(|s| matches!(s.ir, crate::ir_defs::IR::RomInit { .. }));
        stats.advice + stats.lookup + if tables { crate::lower::memory::cost::TABLE_GROUP_CELLS as usize } else { 0 }
    };
    for n in [2usize, 3, 4, 5, 8, 16, 32, 64] {
        let values: Vec<i64> = (0..n as i64).map(|j| (j * 37 + 11) % 13 - 6).collect();
        let shapes: Vec<(&str, Vec<Output>)> = vec![
            ("argsort", (0..n as u32).map(Output::Index).collect()),
            ("sort", (0..n as u32).map(Output::Value).collect()),
            ("both", every_output(n as u32)),
            ("payload", (0..n as u32).map(Output::Payload).collect()),
        ];
        for (shape, outputs) in shapes {
            let mut alternatives = vec![Forced::Batcher, Forced::AdjacentSwap, Forced::RankSelectors, Forced::Permutation];
            if shape == "argsort" {
                alternatives.push(Forced::PackedBatcher);
            }
            let best_other = alternatives.iter().map(|&f| cells(&values, &outputs, f)).min().expect("plans");
            let routed = cells(&values, &outputs, Forced::Routed);
            let rule = cells(&values, &outputs, Forced::CostRule(Target::Halo2Base));
            let best = best_other.min(routed);
            assert!(rule * 100 <= best * 105, "n = {n} {shape}: rule {rule} cells, best {best}");
            let sort = sort_graph(&values, &outputs, Forced::CostRule(Target::Halo2Base));
            let rule_routes = sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::Hint { kind: HintKind::RoutingBits, .. }));
            if rule_routes {
                assert!(routed < best_other, "n = {n} {shape}: routed {routed} picked, another plan {best_other}");
            }
        }
    }
}
