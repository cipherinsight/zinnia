//! The sort lowering: every plan computes NumPy's stable sort (ties
//! included), the cost rule picks the plans the cost model predicts, and
//! every hinted plan accepts its hints only at their honest values (on the
//! mock here; on halo2-base and Noir in those backends' soundness tests,
//! with the same fixtures).

use std::collections::BTreeSet;

use super::cost::{
    adjacent_swap_pairs, choose_plan, cone_size, merge_exchange_pairs, reverse_adjacent_swap_pairs,
    unit_costs, Demand, Network, Pin, Plan, Run,
};
use crate::lower::Target;
use super::test_support::{
    hints_as_inputs, masked_order_statistic_graph, mock_run, sort_graph, stable_sort,
    tampered_inputs, Forced, Output, SortGraph,
};
use crate::ir_defs::IR;

// ── Networks ─────────────────────────────────────────────────────────────

fn apply(pairs: &[(usize, usize)], keys: &mut [u32]) {
    for &(i, j) in pairs {
        assert!(i < j && j < keys.len());
        if keys[j] < keys[i] {
            keys.swap(i, j);
        }
    }
}

/// The 0-1 principle: a comparator network sorts every input iff it sorts
/// every sequence of zeros and ones.
#[test]
fn merge_exchange_sorts_every_zero_one_input() {
    for n in 0..=14usize {
        let pairs = merge_exchange_pairs(n);
        for mask in 0u32..(1 << n) {
            let mut keys: Vec<u32> = (0..n).map(|i| (mask >> i) & 1).collect();
            apply(&pairs, &mut keys);
            assert!(keys.windows(2).all(|w| w[0] <= w[1]), "n={n} mask={mask:b}");
        }
    }
}

#[test]
fn merge_exchange_sorts_permutations_of_larger_sizes() {
    for n in [20usize, 50, 100, 129] {
        let pairs = merge_exchange_pairs(n);
        for seed in 0..20u64 {
            let mut keys: Vec<u32> = (0..n as u32).collect();
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            for i in (1..n).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                keys.swap(i, (state >> 33) as usize % (i + 1));
            }
            apply(&pairs, &mut keys);
            assert_eq!(keys, (0..n as u32).collect::<Vec<_>>());
        }
    }
}

#[test]
fn comparator_counts_and_cones() {
    let counts: Vec<usize> = [2, 3, 20, 50, 100].iter().map(|&n| merge_exchange_pairs(n).len()).collect();
    assert_eq!(counts, vec![1, 3, 97, 395, 1077]);
    let all: BTreeSet<usize> = (0..100).collect();
    assert_eq!(cone_size(&merge_exchange_pairs(100), 100, &all), 1077);
    // The adjacent-swap network's top output needs only the first pass,
    // and the reversed one's bottom output.
    let top: BTreeSet<usize> = [99].into_iter().collect();
    assert_eq!(cone_size(&adjacent_swap_pairs(100), 100, &top), 99);
    let bottom: BTreeSet<usize> = [0].into_iter().collect();
    assert_eq!(cone_size(&reverse_adjacent_swap_pairs(100), 100, &bottom), 99);
    // Both sort (the 0-1 principle).
    for n in 0..=10usize {
        for pairs in [adjacent_swap_pairs(n), reverse_adjacent_swap_pairs(n)] {
            for mask in 0u32..(1 << n) {
                let mut keys: Vec<u32> = (0..n).map(|i| (mask >> i) & 1).collect();
                apply(&pairs, &mut keys);
                assert!(keys.windows(2).all(|w| w[0] <= w[1]), "n={n} mask={mask:b}");
            }
        }
    }
}

// ── Every plan computes NumPy's stable sort ─────────────────────────────

/// Values with ties (several equal values, equal to the extremes), negative
/// values, and a run of one value.
const TIED: [i64; 9] = [5, -3, 5, 0, -3, 7, 5, 0, -9];

fn every_output(n: u32) -> Vec<Output> {
    (0..n).flat_map(|r| [Output::Value(r), Output::Index(r)]).collect()
}

fn assert_computes_numpy(values: &[i64], outputs: &[Output], forced: Forced) {
    let sort = sort_graph(values, outputs, forced);
    assert!(
        !sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::ArrayOp(_))),
        "a placeholder survived"
    );
    let (ok, got) = mock_run(&sort.graph, &sort.inputs);
    match &sort.expected {
        Some(expected) => {
            assert!(ok, "{forced:?} {outputs:?}: honest run rejected");
            assert_eq!(&got, expected, "{forced:?} {outputs:?}");
        }
        None => assert!(!ok, "{forced:?} {outputs:?}: an out-of-range rank was accepted"),
    }
}

#[test]
fn every_plan_is_the_stable_sort_with_ties() {
    let n = TIED.len() as u32;
    for forced in [Forced::Batcher, Forced::AdjacentSwap, Forced::Permutation, Forced::Runs, Forced::Routed] {
        assert_computes_numpy(&TIED, &every_output(n), forced);
        // Some outputs only: a prefix, a suffix, a middle run, isolated ranks.
        for ranks in [vec![0, 1], vec![7, 8], vec![3, 4, 5], vec![0, 4, 8]] {
            let outputs: Vec<Output> = ranks.iter().map(|&r| Output::Index(r)).collect();
            assert_computes_numpy(&TIED, &outputs, forced);
        }
    }
    for ranks in [vec![0], vec![4], vec![8], vec![2, 3, 6]] {
        let outputs: Vec<Output> = ranks.iter().map(|&r| Output::Value(r)).collect();
        assert_computes_numpy(&TIED, &outputs, Forced::Counting);
    }
    for ranks in [vec![0], vec![8], vec![0, 8]] {
        let outputs: Vec<Output> = ranks.iter().map(|&r| Output::Value(r)).collect();
        assert_computes_numpy(&TIED, &outputs, Forced::Extremes);
    }
    for target in [Target::Halo2Base, Target::Noir] {
        assert_computes_numpy(&TIED, &every_output(n), Forced::CostRule(target));
    }
}

/// A payload gathered by the stable argsort (`b[np.argsort(a)]`) is
/// carried by every stable network and read at the pinned indices of every
/// hinted plan: NumPy's gather, ties in original order, alone or with the
/// sort's other outputs, every rank or some.
#[test]
fn payloads_are_gathered_in_stable_order() {
    let n = TIED.len() as u32;
    let every_payload: Vec<Output> = (0..n).map(Output::Payload).collect();
    let some: Vec<Output> = [0, 4, 8].map(Output::Payload).to_vec();
    let mixed = vec![Output::Payload(2), Output::Index(2), Output::Value(5), Output::Payload(7)];
    for outputs in [&every_payload, &some, &mixed] {
        for forced in [Forced::Batcher, Forced::AdjacentSwap, Forced::Permutation, Forced::Runs, Forced::Routed] {
            assert_computes_numpy(&TIED, outputs, forced);
        }
        for target in [Target::Halo2Base, Target::Noir] {
            assert_computes_numpy(&TIED, outputs, Forced::CostRule(target));
        }
    }
    assert_computes_numpy(&TIED, &every_payload, Forced::PackedBatcher);
    assert_computes_numpy(&TIED, &some, Forced::PackedBatcher);
    for n in 1..=4u32 {
        let values: Vec<i64> = TIED[..n as usize].to_vec();
        let outputs: Vec<Output> = (0..n).map(Output::Payload).collect();
        for forced in [Forced::AdjacentSwap, Forced::Permutation, Forced::CostRule(Target::Noir)] {
            assert_computes_numpy(&values, &outputs, forced);
        }
    }
}

/// The packed-key network yields the stable argsort (ties in original
/// order) for every rank, some ranks, runtime ranks, every small length
/// and window-extreme values.
#[test]
fn packed_network_is_the_stable_argsort() {
    let indices = |ranks: &[u32]| ranks.iter().map(|&r| Output::Index(r)).collect::<Vec<_>>();
    let n = TIED.len() as u32;
    assert_computes_numpy(&TIED, &indices(&(0..n).collect::<Vec<_>>()), Forced::PackedBatcher);
    for ranks in [vec![0], vec![8], vec![7, 8], vec![0, 4, 8]] {
        assert_computes_numpy(&TIED, &indices(&ranks), Forced::PackedBatcher);
    }
    for r in [0, 4, 8, -1, 9] {
        assert_computes_numpy(&TIED, &[Output::IndexAt(r)], Forced::PackedBatcher);
    }
    let extremes = [i64::MAX, i64::MIN, 0, i64::MAX, -1, i64::MIN];
    assert_computes_numpy(&extremes, &indices(&[0, 1, 2, 3, 4, 5]), Forced::PackedBatcher);
    let mut state = 0x2545F4914F6CDD1Du64;
    for n in 1..=17usize {
        let values: Vec<i64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) % 5) as i64 - 2
            })
            .collect();
        assert_computes_numpy(&values, &indices(&(0..n as u32).collect::<Vec<_>>()), Forced::PackedBatcher);
        assert_computes_numpy(&values, &every_output(n as u32), Forced::Routed);
    }
    assert_computes_numpy(&extremes, &every_output(6), Forced::Routed);
}

/// Rank selectors give NumPy's stable sort, argsort and payload gather for
/// every length up to 5 (products for n <= 3, rank equality tests above), on
/// every input with ties of three values, for every output kind, constant
/// and runtime ranks (out of range: no proof).
#[test]
fn rank_selectors_are_the_stable_sort() {
    for n in 1..=5u32 {
        // Payload lanes first: a sort starts at its first lane, after the
        // payload's inputs (`test_support::sort_graph` reads them there).
        let payloads: Vec<Output> = (0..n).map(Output::Payload).collect();
        let mixed: Vec<Output> = payloads.iter().copied().chain(every_output(n)).collect();
        for code in 0..3i64.pow(n) {
            let values: Vec<i64> = (0..n).map(|j| (code / 3i64.pow(j)) % 3 - 1).collect();
            assert_computes_numpy(&values, &every_output(n), Forced::RankSelectors);
            assert_computes_numpy(&values, &mixed, Forced::RankSelectors);
        }
        for r in [0, i64::from(n) - 1, -1, i64::from(n)] {
            assert_computes_numpy(&TIED[..n as usize], &[Output::ValueAt(r), Output::IndexAt(r)], Forced::RankSelectors);
        }
    }
    assert_computes_numpy(&TIED, &every_output(TIED.len() as u32), Forced::RankSelectors);
}

/// The 0358 shape: three elements carrying a payload of nine lines. On Noir
/// the rank selectors (3 comparisons, 9 products per rank) cost less than
/// the network's 27 payload exchanges.
#[test]
fn a_small_sort_carrying_a_wide_payload_uses_rank_selectors_on_noir() {
    let mut demand = demand(3, &[], &[], 0);
    demand.payloads = vec![(0..3).collect(); 9];
    assert_eq!(choose_plan(&demand, &unit_costs(Target::Noir)).0, Plan::RankSelectors);
}

#[test]
fn runtime_ranks_are_the_stable_sort_and_out_of_range_has_no_proof() {
    for forced in [Forced::Batcher, Forced::Permutation, Forced::Runs, Forced::Routed] {
        for r in [0, 3, 4, 8] {
            assert_computes_numpy(&TIED, &[Output::ValueAt(r), Output::IndexAt(r)], forced);
        }
        for r in [-1, 9, 1 << 40] {
            assert_computes_numpy(&TIED, &[Output::ValueAt(r)], forced);
            assert_computes_numpy(&TIED, &[Output::IndexAt(r)], forced);
        }
    }
}

#[test]
fn every_plan_agrees_on_pseudo_random_inputs() {
    let mut state = 0x9E3779B97F4A7C15u64;
    for n in [1usize, 2, 5, 12] {
        for _ in 0..4 {
            let values: Vec<i64> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) % 7) as i64 - 3 // many ties
                })
                .collect();
            for forced in [Forced::Batcher, Forced::Permutation, Forced::Runs, Forced::Routed] {
                assert_computes_numpy(&values, &every_output(n as u32), forced);
            }
        }
    }
}

// ── The cost rule ───────────────────────────────────────────────────────

fn demand(n: usize, values: &[usize], indices: &[usize], runtime_values: usize) -> Demand {
    Demand {
        n,
        values: values.iter().copied().collect(),
        indices: indices.iter().copied().collect(),
        runtime_values,
        runtime_indices: 0,
        payloads: Vec::new(),
        masked_extremes: Default::default(),
    }
}

#[test]
fn cost_rule_picks_by_target_and_live_outputs() {
    let halo2 = unit_costs(Target::Halo2Base);
    let noir = unit_costs(Target::Noir);
    let every: Vec<usize> = (0..100).collect();
    // A full argsort of 100: hinted keys checked by a grand product on
    // halo2-base, even paying the challenge's one-time cost (a read is a
    // one-hot selection over 100 slots, a comparator network has 1,077
    // comparisons, the routed network 573 switches), the permutation check
    // on Noir.
    let full = demand(100, &[], &every, 0);
    assert_eq!(choose_plan(&full, &halo2).0, Plan::GrandProduct);
    // With the sorted values live too: the same, carrying the values.
    let both = demand(100, &every, &every, 0);
    assert_eq!(choose_plan(&both, &halo2).0, Plan::GrandProduct);
    assert_eq!(
        choose_plan(&full, &noir).0,
        Plan::Hinted { runs: vec![Run { first: 0, last: 99, pin: Pin::Keys }] }
    );
    // The indices of the 7 largest: one pinned run on Noir; on halo2-base
    // each pinned key reads 100 slots by selection, and the whole sort by a
    // grand product costs less.
    let top: Vec<usize> = (93..100).collect();
    assert_eq!(
        choose_plan(&demand(100, &[], &top, 0), &noir).0,
        Plan::Hinted { runs: vec![Run { first: 93, last: 99, pin: Pin::Keys }] }
    );
    assert_eq!(choose_plan(&demand(100, &[], &top, 0), &halo2).0, Plan::GrandProduct);
    // Two order statistics at runtime ranks (a percentile): counts.
    for costs in [&halo2, &noir] {
        assert_eq!(choose_plan(&demand(20, &[], &[], 2), costs).0, Plan::Hinted { runs: vec![] });
    }
    // The median of 101 values: one count.
    assert_eq!(
        choose_plan(&demand(101, &[50], &[], 0), &halo2).0,
        Plan::Hinted { runs: vec![Run { first: 50, last: 50, pin: Pin::Counts }] }
    );
    // An argsort of three: a network on halo2-base, three native reads and
    // two links on Noir.
    let three = demand(3, &[], &[0, 1, 2], 0);
    assert!(matches!(choose_plan(&three, &halo2).0, Plan::Network { .. } | Plan::Routed));
    assert_eq!(
        choose_plan(&three, &noir).0,
        Plan::Hinted { runs: vec![Run { first: 0, last: 2, pin: Pin::Keys }] }
    );
    // A maximum: a scan of 5 values on halo2-base, the extreme pin for 48
    // values there and for any length on Noir.
    let max_of = |n: usize| demand(n, &[n - 1], &[], 0);
    assert_eq!(choose_plan(&max_of(5), &halo2).0, Plan::Network { network: Network::AdjacentSwapValues });
    let pinned = |n: usize| Plan::Hinted { runs: vec![Run { first: n - 1, last: n - 1, pin: Pin::Extreme }] };
    assert_eq!(choose_plan(&max_of(48), &halo2).0, pinned(48));
    for n in [2, 5, 48] {
        assert_eq!(choose_plan(&max_of(n), &noir).0, pinned(n));
    }
}

// ── Pins: a malicious prover's hints are rejected ───────────────────────

/// Every hint of the lowered graph, replaced by a free input, is accepted
/// at its honest value and rejected at each of `wrong(honest)`.
fn assert_pinned(values: &[i64], outputs: &[Output], forced: Forced, wrong: impl Fn(i64) -> Vec<i64>) {
    let sort = sort_graph(values, outputs, forced);
    let (graph, sites) = hints_as_inputs(&sort);
    assert!(!sites.is_empty(), "{forced:?}: no hint to tamper with");
    let honest = tampered_inputs(&sort, &sites, usize::MAX, 0);
    let (ok, got) = mock_run(&graph, &honest);
    assert!(ok, "{forced:?}: honest hints rejected");
    assert_eq!(Some(got), sort.expected);
    for (k, site) in sites.iter().enumerate() {
        for value in wrong(site.honest) {
            if value == site.honest {
                continue;
            }
            let (ok, got) = mock_run(&graph, &tampered_inputs(&sort, &sites, k, value));
            assert!(!ok, "{forced:?}: hint {k} = {value} (honest {}) accepted, outputs {got:?}", site.honest);
        }
    }
}

/// Wrong argsort lanes: every other index, one past either end, and far
/// out of range (a field element above p/2 reads as negative).
fn wrong_indices(n: i64) -> impl Fn(i64) -> Vec<i64> {
    move |_| (-1..=n).chain([1 << 40, -(1 << 40)]).collect()
}

/// Wrong order statistics: every element value, their neighbours, values
/// between and beyond them.
fn wrong_values(values: &[i64]) -> impl Fn(i64) -> Vec<i64> + '_ {
    move |_| values.iter().flat_map(|&v| [v - 1, v, v + 1]).chain([i64::MIN, i64::MAX]).collect()
}

#[test]
fn permutation_check_rejects_every_other_index() {
    // Wrong lanes include the tie partner of every tied element (an
    // unstable tie order) and every duplicate of another lane.
    let n = TIED.len() as i64;
    assert_pinned(&TIED, &every_output(n as u32), Forced::Permutation, wrong_indices(n));
}

#[test]
fn runs_reject_every_other_index() {
    let n = TIED.len() as i64;
    for ranks in [vec![0u32, 1], vec![6, 7, 8], vec![3, 4], vec![2, 6]] {
        let outputs: Vec<Output> = ranks.iter().map(|&r| Output::Index(r)).collect();
        assert_pinned(&TIED, &outputs, Forced::Runs, wrong_indices(n));
    }
}

#[test]
fn counts_reject_every_other_value() {
    for r in [0u32, 1, 4, 7, 8] {
        assert_pinned(&TIED, &[Output::Value(r)], Forced::Counting, wrong_values(&TIED));
    }
    for r in [0i64, 2, 5, 8] {
        assert_pinned(&TIED, &[Output::ValueAt(r)], Forced::Runs, wrong_values(&TIED));
    }
    for r in [0u32, 8] {
        assert_pinned(&TIED, &[Output::Value(r)], Forced::Extremes, wrong_values(&TIED));
    }
}

/// An extreme whose value the program compares with every element takes
/// its membership from those equality bits: NumPy's mask, ties included,
/// and every other hinted value rejected.
#[test]
fn extremes_with_equality_bits_are_pinned() {
    let n = TIED.len() as u32;
    for r in [0, n - 1] {
        let outputs = [Output::EqualsEach(r)];
        assert_computes_numpy(&TIED, &outputs, Forced::MaskedExtremes);
        let sort = sort_graph(&TIED, &outputs, Forced::MaskedExtremes);
        assert!(
            !sort.graph.stmts.iter().any(|s| matches!(s.ir, IR::MulI)),
            "membership by the equality bits, not a product"
        );
        assert_pinned(&TIED, &outputs, Forced::MaskedExtremes, wrong_values(&TIED));
        for target in [Target::Halo2Base, Target::Noir] {
            assert_computes_numpy(&TIED, &outputs, Forced::CostRule(target));
        }
    }
}

#[test]
fn runtime_rank_index_rejects_every_other_index() {
    let n = TIED.len() as i64;
    for r in [0i64, 1, 4, 8] {
        assert_pinned(&TIED, &[Output::IndexAt(r)], Forced::Runs, wrong_indices(n));
    }
}

/// The two equal elements of a tie, swapped: the order a non-stable sort
/// could give. Every hinted plan rejects it.
#[test]
fn unstable_tie_order_is_rejected() {
    let values = [4, 1, 4, 4];
    let (_, argsort) = stable_sort(&values);
    assert_eq!(argsort, vec![1, 0, 2, 3]);
    for forced in [Forced::Permutation, Forced::Runs] {
        let sort = sort_graph(&values, &every_output(4), forced);
        let (graph, sites) = hints_as_inputs(&sort);
        // Lanes 1 and 2 hold the tied elements 0 and 2; give lane 1 the
        // later one and lane 2 the earlier one.
        let mut inputs = tampered_inputs(&sort, &sites, usize::MAX, 0);
        let lane_of = |rank: i64| sites.iter().position(|s| s.honest == argsort[rank as usize]).unwrap();
        let (a, b) = (lane_of(1), lane_of(2));
        let (ia, ib) = (inputs.len() - sites.len() + a, inputs.len() - sites.len() + b);
        let (va, vb) = (inputs[ia].1, inputs[ib].1);
        inputs[ia].1 = vb;
        inputs[ib].1 = va;
        assert!(!mock_run(&graph, &inputs).0, "{forced:?}: unstable tie order accepted");
    }
}

// ── Order statistics of runtime-length arrays ────────────────────────────

const ACTIVE: [bool; 9] = [true, false, true, true, false, true, true, false, true];

#[test]
fn masked_order_statistics_are_the_active_elements_ranks() {
    for rank in -1..=7 {
        let sort = masked_order_statistic_graph(&TIED, &ACTIVE, rank);
        let (ok, got) = mock_run(&sort.graph, &sort.inputs);
        match &sort.expected {
            Some(expected) => {
                assert!(ok, "rank {rank}: honest run rejected");
                assert_eq!(&got, expected, "rank {rank}");
            }
            None => assert!(!ok, "rank {rank}: a rank beyond the active elements was accepted"),
        }
    }
    // No active element: no proof at any rank.
    let none = [false; 9];
    assert!(!mock_run(&masked_order_statistic_graph(&TIED, &none, 0).graph, &masked_order_statistic_graph(&TIED, &none, 0).inputs).0);
}

#[test]
fn masked_order_statistics_reject_every_other_value() {
    for rank in [0i64, 2, 5] {
        let sort: SortGraph = masked_order_statistic_graph(&TIED, &ACTIVE, rank);
        let (graph, sites) = hints_as_inputs(&sort);
        assert_eq!(sites.len(), 1);
        let honest = tampered_inputs(&sort, &sites, usize::MAX, 0);
        assert!(mock_run(&graph, &honest).0);
        // Every element value (padding ones included), their neighbours.
        for value in TIED.iter().flat_map(|&v| [v - 1, v, v + 1]) {
            if value != sites[0].honest {
                let inputs = tampered_inputs(&sort, &sites, 0, value);
                assert!(!mock_run(&graph, &inputs).0, "rank {rank}: {value} accepted");
            }
        }
    }
}

