//! Tests of the array claims (`array_facts`): implication, bases, facts by
//! definition, the merged range checks, and the sort of a sorted input.

use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::circuit_input::InputPath;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::array_facts::{self, basis};
use super::emit::Options;
use super::graph::{Bound, Claim, ListEntry, ListId};
use super::sort::test_support::mock_run;
use super::{OpKind, Target};

fn stmts(ids: &[StmtId]) -> ListId {
    ListId::of_stmts(ids)
}

#[test]
fn array_claims_imply_their_parts_and_their_values_claims() {
    ListId::reset();
    array_facts::set_enabled(true);
    let all = stmts(&[10, 11, 12, 13]);
    let within = Claim::ElementsWithin { list: all, lo: Some(0), hi: Some(255) };
    assert!(within.implies(&Claim::Within { x: 12, lo: Some(-5), hi: Some(255) }));
    assert!(within.implies(&Claim::Within { x: 12, lo: None, hi: Some(300) }));
    assert!(within.implies(&Claim::InRange { x: 10, lo: 0, hi: Bound::Const(256) }));
    assert!(within.implies(&Claim::NonNegative(13)));
    assert!(within.implies(&Claim::ElementsWithin { list: stmts(&[13, 10]), lo: None, hi: Some(255) }));
    // Not about the list's values, or wider.
    assert!(!within.implies(&Claim::Within { x: 14, lo: Some(0), hi: None }));
    assert!(!within.implies(&Claim::Within { x: 12, lo: Some(1), hi: None }));
    assert!(!within.implies(&Claim::InRange { x: 10, lo: 0, hi: Bound::Const(255) }));
    assert!(!within.implies(&Claim::NonZero(10)));
    let open = Claim::ElementsWithin { list: all, lo: Some(0), hi: None };
    assert!(!open.implies(&Claim::Within { x: 10, lo: None, hi: Some(i64::MAX) }));

    let sorted = Claim::NonDecreasing(all);
    assert!(sorted.implies(&Claim::NonDecreasing(stmts(&[10, 12, 13]))));
    assert!(sorted.implies(&Claim::Ordered { x: 10, y: 13, strict: false }));
    assert!(!sorted.implies(&Claim::Ordered { x: 10, y: 13, strict: true }));
    assert!(!sorted.implies(&Claim::Ordered { x: 13, y: 10, strict: false }));
    assert!(!sorted.implies(&Claim::NonDecreasing(stmts(&[12, 10]))));
    assert!(!sorted.implies(&Claim::StrictlyIncreasing(all)));
    assert!(!sorted.implies(&Claim::Distinct(all)));
    let increasing = Claim::StrictlyIncreasing(all);
    assert!(increasing.implies(&sorted));
    assert!(increasing.implies(&Claim::Ordered { x: 11, y: 12, strict: true }));
    assert!(increasing.implies(&Claim::Distinct(stmts(&[13, 11]))));
    assert!(!increasing.implies(&Claim::Distinct(stmts(&[13, 13]))));

    let p = stmts(&[20, 21, 22]);
    let permutation = Claim::Permutation(p, ListId::range(3));
    assert!(permutation.implies(&Claim::InRange { x: 21, lo: 0, hi: Bound::Const(3) }));
    assert!(permutation.implies(&Claim::Within { x: 22, lo: Some(i64::MIN), hi: Some(i64::MAX) }));
    assert!(permutation.implies(&Claim::Distinct(stmts(&[22, 20]))));
    assert!(permutation.implies(&Claim::Permutation(ListId::range(3), p)));
    assert!(!permutation.implies(&Claim::InRange { x: 21, lo: 0, hi: Bound::Const(2) }));
    assert!(!permutation.implies(&Claim::NonDecreasing(p)));

    let counted = Claim::CountBound { list: all, max_nonzero: 2 };
    assert!(counted.implies(&Claim::CountBound { list: all, max_nonzero: 3 }));
    assert!(!counted.implies(&Claim::CountBound { list: all, max_nonzero: 1 }));

    // Off (`ZINNIA_ABLATE_ARRAY_FACTS`): a claim implies only itself.
    array_facts::set_enabled(false);
    assert!(!within.implies(&Claim::NonNegative(13)));
    assert!(!sorted.implies(&Claim::Ordered { x: 10, y: 13, strict: false }));
    assert!(!permutation.implies(&Claim::Distinct(p)));
    assert!(permutation.implies(&permutation));
    array_facts::set_enabled(true);
}

#[test]
fn array_claims_decompose_into_their_values_claims() {
    ListId::reset();
    let list = ListId::intern(&[ListEntry::Int(1), ListEntry::Stmt(5), ListEntry::Stmt(6), ListEntry::Int(9)]);
    assert_eq!(
        basis(&Claim::NonDecreasing(list)),
        Some(vec![
            Claim::Within { x: 5, lo: Some(1), hi: None },
            Claim::Ordered { x: 5, y: 6, strict: false },
            Claim::Within { x: 6, lo: None, hi: Some(9) },
        ])
    );
    assert_eq!(
        basis(&Claim::StrictlyIncreasing(list)),
        Some(vec![
            Claim::Within { x: 5, lo: Some(2), hi: None },
            Claim::Ordered { x: 5, y: 6, strict: true },
            Claim::Within { x: 6, lo: None, hi: Some(8) },
        ])
    );
    // A constant pair out of order: no basis.
    let unordered = ListId::intern(&[ListEntry::Int(3), ListEntry::Int(2)]);
    assert_eq!(basis(&Claim::NonDecreasing(unordered)), None);
    assert_eq!(
        basis(&Claim::ElementsWithin { list, lo: Some(0), hi: Some(10) }),
        Some(vec![
            Claim::Within { x: 5, lo: Some(0), hi: None },
            Claim::Within { x: 5, lo: None, hi: Some(10) },
            Claim::Within { x: 6, lo: Some(0), hi: None },
            Claim::Within { x: 6, lo: None, hi: Some(10) },
        ])
    );
    assert_eq!(basis(&Claim::ElementsWithin { list, lo: Some(2), hi: None }), None);
}

/// Inputs `a0..a{n-1}` with their int64 window checks, as the IR generator
/// reads them.
pub(crate) fn read_inputs(b: &mut IRBuilder, prefix: &str, n: usize) -> Vec<Value> {
    (0..n)
        .map(|j| {
            let x = b.ir_read_integer(InputPath::new(format!("{prefix}{j}"), vec![]), false);
            let (lo, hi) = (b.ir_constant_int(i64::MIN), b.ir_constant_int(i64::MAX));
            let above = b.ir_greater_than_or_equal_i(&x, &lo);
            b.ir_assert(&above);
            let below = b.ir_less_than_or_equal_i(&x, &hi);
            b.ir_assert(&below);
            x
        })
        .collect()
}

/// `graph` lowered as the compiler does, with array claims on or off.
pub(crate) fn lowered(graph: &IRGraph, target: Target, no_array_facts: bool) -> IRGraph {
    let options = Options { no_array_facts, ..Options::default() };
    super::emit::lower_rounds_with(graph.clone(), target, &super::registry(), &options)
}

pub(crate) fn count(graph: &IRGraph, f: impl Fn(&IR) -> bool) -> usize {
    graph.stmts.iter().filter(|s| f(&s.ir)).count()
}

/// `forall(x, in_range(lo, hi))` over inputs `x0..`: each value's window
/// checks and the predicate's two checks, then every value exposed and
/// compared with `pivot`.
pub(crate) fn ranged_program(n: usize, lo: i64, hi: i64, pivot: i64) -> IRGraph {
    let mut b = IRBuilder::new();
    let xs = read_inputs(&mut b, "x", n);
    for x in &xs {
        let (l, h) = (b.ir_constant_int(lo), b.ir_constant_int(hi));
        let above = b.ir_greater_than_or_equal_i(x, &l);
        b.ir_assert(&above);
        let below = b.ir_less_than_or_equal_i(x, &h);
        b.ir_assert(&below);
    }
    for x in &xs {
        b.ir_expose_public_i(x);
        let p = b.ir_constant_int(pivot);
        let less = b.ir_less_than_i(x, &p);
        let less = b.ir_int_cast(&less);
        b.ir_expose_public_i(&less);
    }
    b.export_ir_graph()
}

#[test]
fn a_range_checked_on_both_sides_is_one_narrow_check() {
    for (lo, hi) in [(0, 255), (-100, 27), (5, 5), (1000, 1999)] {
        let program = ranged_program(3, lo, hi, (lo + hi) / 2);
        for target in [Target::Halo2Base, Target::Noir] {
            let merged = lowered(&program, target, false);
            assert_eq!(count(&merged, |ir| matches!(ir, IR::AssertU128Lt)), 3, "{lo}..{hi} {target:?}");
            assert_eq!(count(&merged, |ir| matches!(ir, IR::Assert)), 0, "{lo}..{hi} {target:?}");
            let kept = lowered(&program, target, true);
            assert_eq!(count(&kept, |ir| matches!(ir, IR::Assert)), 12, "{lo}..{hi} {target:?} (switch off)");
            for v in [lo - 1, lo, (lo + hi) / 2, hi, hi + 1, i64::MIN, i64::MAX, lo - (1 << 40), hi + (1 << 40)] {
                let inputs: Vec<(String, i64)> = (0..3).map(|j| (format!("x{j}"), v)).collect();
                let (ok, out) = mock_run(&merged, &inputs);
                let (ok_kept, out_kept) = mock_run(&kept, &inputs);
                assert_eq!(ok, lo <= v && v <= hi, "{lo}..{hi}: x = {v}");
                assert_eq!(ok, ok_kept, "{lo}..{hi}: x = {v}");
                if ok {
                    assert_eq!(out, out_kept);
                }
            }
        }
    }
}

/// A sort of inputs `a0..` that the program asserts non-decreasing (as
/// `@requires(is_sorted(a))` does), with the minimum, the maximum, the value
/// and index of rank 2, a runtime-rank value and index (input `r`), exposed.
fn sorted_program(n: usize) -> IRGraph {
    let mut b = IRBuilder::new();
    let a = read_inputs(&mut b, "a", n);
    for pair in a.windows(2) {
        let ordered = b.ir_less_than_or_equal_i(&pair[0], &pair[1]);
        b.ir_assert(&ordered);
    }
    let r = b.ir_read_integer(InputPath::new("r", vec![]), false);
    let elements: Arc<[Value]> = a.into();
    let outputs = [
        b.ir_sorted(&elements, 0, false),
        b.ir_sorted(&elements, n as u32 - 1, false),
        b.ir_sorted(&elements, 2, false),
        b.ir_sorted(&elements, 2, true),
        b.ir_sorted_at_runtime_rank(&elements, &r, false),
        b.ir_sorted_at_runtime_rank(&elements, &r, true),
    ];
    for out in outputs {
        b.ir_expose_public_i(&out);
    }
    b.export_ir_graph()
}

#[test]
fn a_sort_of_a_sorted_input_reads_its_input() {
    let n = 5;
    let program = sorted_program(n);
    for target in [Target::Halo2Base, Target::Noir] {
        let lowered_graph = lowered(&program, target, false);
        assert_eq!(count(&lowered_graph, |ir| matches!(ir, IR::Hint { .. } | IR::KeyLess { .. })), 0, "{target:?}");
        let sorted_by_layer = lowered(&program, target, true);
        assert!(count(&sorted_by_layer, |ir| matches!(ir, IR::Hint { .. } | IR::KeyLess { .. })) > 0);
        for values in [[1, 2, 3, 4, 5], [7, 7, 7, 7, 7], [-3, 0, 0, 9, 9], [1, 3, 2, 4, 5], [5, 4, 3, 2, 1]] {
            let sorted = values.windows(2).all(|w| w[0] <= w[1]);
            for r in [-1, 0, 3, 4, 5] {
                let mut inputs: Vec<(String, i64)> =
                    values.iter().enumerate().map(|(j, &v)| (format!("a{j}"), v)).collect();
                inputs.push(("r".to_string(), r));
                let (ok, out) = mock_run(&lowered_graph, &inputs);
                assert_eq!(ok, sorted && (0..n as i64).contains(&r), "{values:?} r = {r}");
                if ok {
                    let r = r as usize;
                    assert_eq!(out, vec![values[0], values[n - 1], values[2], 2, values[r], r as i64], "{values:?}");
                }
            }
        }
    }
}

#[test]
fn facts_by_definition() {
    ListId::reset();
    array_facts::set_enabled(true);
    let mut b = IRBuilder::new();
    let a = read_inputs(&mut b, "a", 3);
    let elements: Arc<[Value]> = a.clone().into();
    let lanes: Vec<Value> = (0..3).map(|r| b.ir_sorted(&elements, r, false)).collect();
    let index_lanes: Vec<Value> = (0..3).map(|r| b.ir_sorted(&elements, r, true)).collect();
    let square = b.ir_mul_i(&a[0], &a[0]);
    let abs = b.ir_abs_i(&a[1]);
    let sum = b.ir_add_i(&square, &abs);
    let difference = b.ir_sub_i(&square, &abs);
    let cube = b.ir_mul_i(&square, &a[0]);
    let graph = b.export_ir_graph();
    let id = |v: &Value| v.stmt_id().unwrap();
    let lane_ids: Vec<StmtId> = lanes.iter().map(id).collect();
    let index_ids: Vec<StmtId> = index_lanes.iter().map(id).collect();
    let definitional = |claim: Claim| array_facts::definitional(&graph, &claim);
    assert!(definitional(Claim::NonDecreasing(stmts(&lane_ids))));
    assert!(definitional(Claim::NonDecreasing(stmts(&[lane_ids[0], lane_ids[2]]))));
    assert!(!definitional(Claim::NonDecreasing(stmts(&[lane_ids[2], lane_ids[0]]))));
    assert!(!definitional(Claim::StrictlyIncreasing(stmts(&lane_ids))));
    assert!(!definitional(Claim::NonDecreasing(stmts(&[id(&a[0]), id(&a[1])]))));
    assert!(definitional(Claim::Distinct(stmts(&index_ids))));
    assert!(definitional(Claim::Permutation(stmts(&[index_ids[2], index_ids[0], index_ids[1]]), ListId::range(3))));
    assert!(!definitional(Claim::Permutation(stmts(&index_ids[..2]), ListId::range(2))));
    assert!(definitional(Claim::NonNegative(id(&square))));
    assert!(definitional(Claim::NonNegative(id(&sum))));
    assert!(!definitional(Claim::NonNegative(id(&difference))));
    assert!(!definitional(Claim::NonNegative(id(&cube))));
    assert!(!definitional(Claim::NonNegative(id(&a[0]))));
    let _ = OpKind::Sort;
}

// ── Fixtures shared with the backends' tests ────────────────────────────

/// Asserts `x0 <= x1 <= ...` as `@requires(is_sorted(x))` does.
pub(crate) fn assert_sorted(b: &mut IRBuilder, xs: &[Value]) {
    for pair in xs.windows(2) {
        let ordered = b.ir_less_than_or_equal_i(&pair[0], &pair[1]);
        b.ir_assert(&ordered);
    }
}

/// Asserts `lo <= x <= hi` for every `x`, as `forall(x, in_range(lo, hi))`
/// does.
pub(crate) fn assert_in_range(b: &mut IRBuilder, xs: &[Value], lo: i64, hi: i64) {
    for x in xs {
        let (l, h) = (b.ir_constant_int(lo), b.ir_constant_int(hi));
        let above = b.ir_greater_than_or_equal_i(x, &l);
        b.ir_assert(&above);
        let below = b.ir_less_than_or_equal_i(x, &h);
        b.ir_assert(&below);
    }
}

/// `np.argmax` and `np.argmin` of inputs `a0..a{n-1}` asserted sorted
/// (when `sorted`), both exposed, lowered on `target` by the compiler's
/// selection, or with the argmax forced to a count or a hinted position.
pub(crate) fn argextreme_program(n: usize, sorted: bool, forced_hint: Option<bool>, target: Target) -> IRGraph {
    let mut b = IRBuilder::new();
    let a = read_inputs(&mut b, "a", n);
    if sorted {
        assert_sorted(&mut b, &a);
    }
    for max in [true, false] {
        let lane = super::sort::arg_extreme::lane(&mut b, &a, max, false);
        b.ir_expose_public_i(&lane);
    }
    let graph = b.export_ir_graph();
    match forced_hint {
        None => lowered(&graph, target, false),
        Some(hinted) => {
            let forced = super::sort::arg_extreme::ForcedSortedArgmax { hinted };
            let first = super::emit::lower(&graph, target, &[&forced]).expect("lanes to lower");
            super::emit::lower_rounds(first, target, &super::registry())
        }
    }
}

/// NumPy's argmax and argmin (first extreme).
pub(crate) fn numpy_argextremes(values: &[i64]) -> Vec<i64> {
    let max = values.iter().max().unwrap();
    let min = values.iter().min().unwrap();
    vec![values.iter().position(|v| v == max).unwrap() as i64, values.iter().position(|v| v == min).unwrap() as i64]
}

/// Every non-decreasing sequence of `n` values in `0..k` (ties included).
pub(crate) fn sorted_sequences(n: usize, k: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|prefix: Vec<i64>| {
                let start = *prefix.last().unwrap_or(&0);
                (start..k).map(move |v| {
                    let mut next = prefix.clone();
                    next.push(v);
                    next
                })
            })
            .collect();
    }
    out
}

fn inputs_of(prefix: &str, values: &[i64]) -> Vec<(String, i64)> {
    values.iter().enumerate().map(|(j, &v)| (format!("{prefix}{j}"), v)).collect()
}

#[test]
fn the_argmax_of_a_sorted_line_is_numpys_first_maximum() {
    for n in 2..=5 {
        for forced in [None, Some(false), Some(true)] {
            let graph = argextreme_program(n, true, forced, Target::Halo2Base);
            for values in sorted_sequences(n, 3) {
                let (ok, out) = mock_run(&graph, &inputs_of("a", &values));
                assert!(ok, "{values:?} {forced:?}");
                assert_eq!(out, numpy_argextremes(&values), "{values:?} {forced:?}");
            }
            // An unsorted line is rejected (the order checks the claim rests
            // on are kept).
            let mut unsorted: Vec<i64> = (0..n as i64).collect();
            unsorted.swap(0, 1);
            assert!(!mock_run(&graph, &inputs_of("a", &unsorted)).0, "{forced:?}");
        }
        // No order claim: the running selects, on any line.
        let graph = argextreme_program(n, false, None, Target::Halo2Base);
        for values in [vec![3; n], (0..n as i64).rev().collect(), (0..n as i64).map(|j| (j * 7) % 5).collect()] {
            let (ok, out) = mock_run(&graph, &inputs_of("a", &values));
            assert!(ok);
            assert_eq!(out, numpy_argextremes(&values), "{values:?}");
        }
    }
}

#[test]
fn the_hinted_argmax_position_is_pinned() {
    for values in sorted_sequences(4, 3) {
        let graph = argextreme_program(4, true, Some(true), Target::Halo2Base);
        let graph = super::sort::search::hint_as_input(&graph);
        let expected = numpy_argextremes(&values)[0];
        for u in (-2..=5).chain([1 << 40, -(1 << 40)]) {
            let mut inputs = inputs_of("a", &values);
            inputs.push(("hint".to_string(), u));
            let (ok, out) = mock_run(&graph, &inputs);
            assert_eq!(ok, u == expected, "{values:?}: position {u}");
            if ok {
                assert_eq!(out[0], expected);
            }
        }
    }
}

/// `np.clip(x, lo, hi)` (its `min(max(x, lo), hi)`) and `np.maximum(x, lo)`
/// of inputs checked into `[lo, hi]`, exposed.
pub(crate) fn clip_program(n: usize, lo: i64, hi: i64) -> IRGraph {
    let mut b = IRBuilder::new();
    let xs = read_inputs(&mut b, "x", n);
    assert_in_range(&mut b, &xs, lo, hi);
    for x in &xs {
        let (l, h) = (b.ir_constant_int(lo), b.ir_constant_int(hi));
        let above = b.ir_greater_than_i(x, &l);
        let raised = b.ir_select_i(&above, x, &l);
        let below = b.ir_less_than_i(&raised, &h);
        let clipped = b.ir_select_i(&below, &raised, &h);
        b.ir_expose_public_i(&clipped);
        // `x < hi + 1` is decided true.
        let past = b.ir_constant_int(hi + 1);
        let inside = b.ir_less_than_i(x, &past);
        let inside = b.ir_int_cast(&inside);
        b.ir_expose_public_i(&inside);
    }
    b.export_ir_graph()
}

#[test]
fn a_clip_of_a_range_checked_value_is_the_value() {
    for (lo, hi) in [(0, 255), (-7, 9)] {
        let program = clip_program(3, lo, hi);
        for target in [Target::Halo2Base, Target::Noir] {
            let graph = lowered(&program, target, false);
            assert_eq!(count(&graph, |ir| matches!(ir, IR::SelectI | IR::LtI | IR::GtI)), 0, "{target:?}");
            let kept = lowered(&program, target, true);
            assert!(count(&kept, |ir| matches!(ir, IR::SelectI)) > 0);
            for v in [lo - 1, lo, lo + 1, hi - 1, hi, hi + 1] {
                let inputs = inputs_of("x", &[v, v, v]);
                let (ok, out) = mock_run(&graph, &inputs);
                assert_eq!(ok, lo <= v && v <= hi, "{v}");
                assert_eq!(ok, mock_run(&kept, &inputs).0);
                if ok {
                    assert_eq!(out, [v, 1, v, 1, v, 1].to_vec());
                }
            }
        }
    }
}

/// `np.max` of inputs checked into `[lo, hi]`, lowered by the extreme pin
/// with its hinted value bounded into the range; its one hint.
pub(crate) fn ranged_max_program(n: usize, lo: i64, hi: i64) -> IRGraph {
    assert!(n >= 16, "the extreme pin is chosen for long lines");
    let mut b = IRBuilder::new();
    let a = read_inputs(&mut b, "a", n);
    assert_in_range(&mut b, &a, lo, hi);
    let elements: Arc<[Value]> = a.into();
    let max = b.ir_sorted(&elements, n as u32 - 1, false);
    b.ir_expose_public_i(&max);
    lowered(&b.export_ir_graph(), Target::Halo2Base, false)
}

#[test]
fn a_bounded_hint_of_a_ranged_maximum_is_pinned() {
    let graph = ranged_max_program(16, -3, 20);
    // The hint's own narrow check and the merged checks of the elements.
    assert_eq!(count(&graph, |ir| matches!(ir, IR::Hint { .. })), 1);
    assert_eq!(count(&graph, |ir| matches!(ir, IR::AssertU128Lt)), 17);
    let hinted = super::sort::search::hint_as_input(&graph);
    for values in [line16(&[3, -3, 20, 7]), line16(&[-3]), line16(&[5, 9, 9, 1])] {
        let expected = *values.iter().max().unwrap();
        for v in [-4, -3, expected - 1, expected, expected + 1, 20, 21, 1 << 40] {
            let mut inputs = inputs_of("a", &values);
            inputs.push(("hint".to_string(), v));
            let (ok, out) = mock_run(&hinted, &inputs);
            assert_eq!(ok, v == expected, "{values:?}: maximum {v}");
            if ok {
                assert_eq!(out, vec![expected]);
            }
        }
    }
}

/// halo2-base: the constructions above on the real circuit (`MockProver`):
/// honest witnesses accepted with NumPy's outputs, every tampered hint and
/// every input outside its checked range rejected.
#[cfg(feature = "halo2-base-backend")]
#[test]
fn fact_constructions_on_halo2_base() {
    use crate::circuit_input::InputNode;
    use crate::prove::halo2_base::synthesizer::two_phase_tests::{mock_run as halo2_run, Tamper};
    use crate::prove::kernel::i64_to_fp;
    let run = |graph: &IRGraph, inputs: &[(String, i64)], tamper: &Tamper| {
        let named: Vec<(&str, InputNode)> = inputs.iter().map(|(k, v)| (k.as_str(), InputNode::Int(*v))).collect();
        halo2_run(graph, &named, tamper)
    };
    let outputs = |run: &crate::prove::halo2_base::synthesizer::two_phase_tests::Run| -> Vec<i64> {
        run.outputs.iter().map(|v| crate::prove::kernel::fp_to_i64(*v)).collect()
    };
    // Merged range checks, with a lower bound away from 0 (the range is
    // carried across the offset to the comparisons after it).
    for (lo, hi) in [(0, 255), (-100, 27), (1000, 1999)] {
        let graph = lowered(&ranged_program(2, lo, hi, (lo + hi) / 2), Target::Halo2Base, false);
        for v in [lo - 1, lo, (lo + hi) / 2, hi, hi + 1] {
            let result = run(&graph, &inputs_of("x", &[v, lo]), &Tamper::default());
            assert_eq!(result.accepted, lo <= v && v <= hi, "{lo}..{hi}: {v}");
            if result.accepted {
                assert_eq!(outputs(&result), vec![v, i64::from(v < (lo + hi) / 2), lo, i64::from(lo < (lo + hi) / 2)]);
            }
        }
    }
    // The hinted argmax of a sorted line, and the bounded maximum's hint.
    let argmax = argextreme_program(4, true, Some(true), Target::Halo2Base);
    for values in [[0, 1, 1, 1], [2, 2, 2, 2], [0, 0, 1, 2]] {
        let honest = run(&argmax, &inputs_of("a", &values), &Tamper::default());
        assert!(honest.accepted, "{values:?}: {} hints", honest.hints);
        assert_eq!(outputs(&honest), numpy_argextremes(&values));
        for delta in [-2, -1, 1, 2] {
            let tampered = Tamper { hints: vec![(0, i64_to_fp(delta))], ..Tamper::default() };
            assert!(!run(&argmax, &inputs_of("a", &values), &tampered).accepted, "{values:?} {delta} ({} hints)", honest.hints);
        }
        assert!(!run(&argmax, &inputs_of("a", &[1, 0, 2, 3]), &Tamper::default()).accepted);
    }
    let max = ranged_max_program(16, -3, 20);
    for values in [line16(&[3, -3, 20, 7]), line16(&[-3])] {
        let honest = run(&max, &inputs_of("a", &values), &Tamper::default());
        assert!(honest.accepted);
        assert_eq!(outputs(&honest), vec![*values.iter().max().unwrap()]);
        for delta in [-1, 1, 1 << 30] {
            let tampered = Tamper { hints: vec![(0, i64_to_fp(delta))], ..Tamper::default() };
            assert!(!run(&max, &inputs_of("a", &values), &tampered).accepted, "{values:?} {delta}");
        }
    }
    // A clip of range-checked values is the values.
    let clip = lowered(&clip_program(2, -7, 9), Target::Halo2Base, false);
    for v in [-8, -7, 0, 9, 10] {
        let result = run(&clip, &inputs_of("x", &[v, v]), &Tamper::default());
        assert_eq!(result.accepted, (-7..=9).contains(&v));
        if result.accepted {
            assert_eq!(outputs(&result), vec![v, 1, v, 1]);
        }
    }
}

/// Sixteen values cycling through `values`.
fn line16(values: &[i64]) -> Vec<i64> {
    (0..16).map(|j| values[j % values.len()]).collect()
}
