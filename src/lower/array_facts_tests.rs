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
fn read_inputs(b: &mut IRBuilder, prefix: &str, n: usize) -> Vec<Value> {
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
fn lowered(graph: &IRGraph, target: Target, no_array_facts: bool) -> IRGraph {
    let options = Options { no_array_facts, ..Options::default() };
    super::emit::lower_rounds_with(graph.clone(), target, &super::registry(), &options)
}

fn count(graph: &IRGraph, f: impl Fn(&IR) -> bool) -> usize {
    graph.stmts.iter().filter(|s| f(&s.ir)).count()
}

/// `forall(x, in_range(lo, hi))` over inputs `x0..`: each value's window
/// checks and the predicate's two checks, then every value exposed and
/// compared with `pivot`.
fn ranged_program(n: usize, lo: i64, hi: i64, pivot: i64) -> IRGraph {
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
