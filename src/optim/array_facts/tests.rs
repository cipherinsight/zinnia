//! Unit tests of the array-fact domain, segmentations and store.

use std::collections::BTreeMap;

use super::*;
use crate::types::StmtId;

/// Intervals of runtime integers, for symbolic bounds.
struct Known(BTreeMap<StmtId, (i64, i64)>);

impl Scalars for Known {
    fn interval(&self, stmt: StmtId) -> Option<(i64, i64)> {
        self.0.get(&stmt).copied()
    }
}

fn known(pairs: &[(StmtId, (i64, i64))]) -> Known {
    Known(pairs.iter().copied().collect())
}

fn stmts(ids: &[StmtId]) -> Vec<Cell> {
    ids.iter().map(|&s| Cell::Stmt(s)).collect()
}

#[test]
fn bounds_compare_by_offset_on_one_statement_and_by_interval_otherwise() {
    let s = known(&[(7, (0, 10))]);
    let count = Bound::stmt(7);
    assert!(count.shifted(-1).unwrap().lt(&count, &s));
    assert!(Bound::Const(-1).lt(&count, &s));
    assert!(!Bound::Const(3).le(&count, &s), "3 <= count is unknown for count in [0, 10]");
    assert!(count.le(&Bound::Const(10), &s));
    assert!(!count.le(&Bound::Const(9), &s));
    assert!(!count.le(&Bound::Const(100), &NoScalars), "no interval: nothing is known");
}

#[test]
fn range_join_keeps_a_common_symbolic_end_and_falls_back_to_constants() {
    let s = known(&[(7, (2, 10))]);
    let below_count = Range { lo: Some(Bound::Const(0)), hi: Some(Bound::Stmt { stmt: 7, offset: -1 }) };
    let joined = below_count.join(&Range::new(0, 1), &s);
    assert_eq!(joined.lo, Some(Bound::Const(0)));
    // 1 <= count - 1 holds (count >= 2), so the symbolic end is the upper one.
    assert_eq!(joined.hi, Some(Bound::Stmt { stmt: 7, offset: -1 }));
    let wide = below_count.join(&Range::new(0, 20), &s);
    assert_eq!(wide.hi, Some(Bound::Const(20)));
    let unknown = below_count.join(&Range::new(0, 5), &s);
    assert_eq!(unknown.hi, Some(Bound::Const(9)), "incomparable ends become the greatest constant");
}

#[test]
fn range_implications() {
    let s = known(&[(7, (1, 4))]);
    let r = Range { lo: Some(Bound::Const(0)), hi: Some(Bound::Stmt { stmt: 7, offset: -1 }) };
    assert!(r.implies_ge(&Bound::Const(0), &s));
    assert!(r.implies_le(&Bound::Stmt { stmt: 7, offset: -1 }, &s), "x <= count - 1, i.e. x < count");
    assert!(r.implies_le(&Bound::Const(3), &s));
    assert!(!r.implies_le(&Bound::Const(2), &s));
    assert!(Range::boolean().is_boolean(&s));
    assert!(!Range::new(0, 2).is_boolean(&s));
}

#[test]
fn interval_transfer_functions() {
    let s = NoScalars;
    let a = Range::new(-3, 5);
    let b = Range::new(2, 4);
    assert_eq!(transfer::add(&a, &b, &s), Range::new(-1, 9));
    assert_eq!(transfer::sub(&a, &b, &s), Range::new(-7, 3));
    assert_eq!(transfer::mul(&a, &b, &s), Range::new(-12, 20));
    assert_eq!(transfer::square(&a, &s), Range::new(0, 25));
    assert_eq!(transfer::square(&Range::new(-5, -2), &s), Range::new(4, 25));
    assert_eq!(transfer::abs(&a, &s), Range::new(0, 5));
    assert_eq!(transfer::neg(&a, &s), Range::new(-5, 3));
    assert_eq!(transfer::min(&a, &b, &s), Range::new(-3, 4));
    assert_eq!(transfer::max(&a, &b, &s), Range::new(2, 5));
    assert_eq!(transfer::floor_div_by(&Range::new(-7, 7), 2, &s), Range::new(-4, 3));
    assert_eq!(transfer::mod_by(5), Range::new(0, 4));
    // A constant operand shifts a symbolic end.
    let r = Range { lo: Some(Bound::Const(0)), hi: Some(Bound::Stmt { stmt: 9, offset: -1 }) };
    assert_eq!(transfer::add(&r, &Range::point(1), &s).hi, Some(Bound::Stmt { stmt: 9, offset: 0 }));
}

#[test]
fn interval_transfer_drops_ends_outside_i64() {
    let s = NoScalars;
    let full = Range::new(i64::MIN, i64::MAX);
    let sum = transfer::add(&full, &full, &s);
    assert!(sum.is_top(), "both ends overflow i64: {sum:?}");
    let sq = transfer::square(&full, &s);
    assert_eq!(sq.lo, Some(Bound::Const(0)));
    assert_eq!(sq.hi, None);
}

#[test]
fn join_drops_order_unless_the_formula_is_the_same() {
    let s = NoScalars;
    let inc = ElemFacts { range: Range::new(0, 9), order: Some(Order::Increasing), affine: None };
    assert_eq!(inc.join(&inc, &s).order, None, "a choice between two increasing arrays need not increase");
    let arange = ElemFacts {
        range: Range::new(0, 9),
        order: Some(Order::Increasing),
        affine: Some(Affine { base: 0, step: 1 }),
    };
    assert_eq!(arange.join(&arange, &s).order, Some(Order::Increasing));
}

#[test]
fn segmentation_locates_positions_against_a_symbolic_cut() {
    // {0} live in [0, 4] {count} padding = 0 {6}, count in [2, 5].
    let s = known(&[(7, (2, 5))]);
    let live = ElemFacts { range: Range::new(0, 4), order: Some(Order::Increasing), affine: None };
    let pad = ElemFacts::range(Range::point(0));
    let seg = Segmentation::live_then_padding(6, Bound::stmt(7), live, pad);
    assert_eq!(seg.at(0, &s), live, "position 0 < 2 <= count: live");
    assert_eq!(seg.at(1, &s), live);
    assert_eq!(seg.at(5, &s), pad, "position 5 >= 5 >= count: padding");
    let middle = seg.at(3, &s);
    assert_eq!(middle.range, Range::new(0, 4), "either segment: the join");
    assert_eq!(middle.order, None);
    assert_eq!(seg.hull(&s).range, Range::new(0, 4));
}

#[test]
fn segmentation_slices_and_reverses_affine_runs() {
    let s = NoScalars;
    let arange = ElemFacts {
        range: Range::new(10, 19),
        order: Some(Order::Increasing),
        affine: Some(Affine { base: 10, step: 1 }),
    };
    let seg = Segmentation::uniform(10, arange);
    assert_eq!(seg.at(3, &s).range, Range::point(13));
    let slice = seg.slice(2, 5, &s);
    assert_eq!(slice.len(), 3);
    assert_eq!(slice.hull(&s).range, Range::new(12, 14));
    assert_eq!(slice.at(0, &s).range, Range::point(12));
    let rev = slice.reversed(&s);
    assert_eq!(rev.at(0, &s).range, Range::point(14));
    assert_eq!(rev.hull(&s).order, Some(Order::Decreasing));
}

#[test]
fn permutation_implies_the_index_range() {
    let s = NoScalars;
    let p = Segmentation::uniform(4, ElemFacts::TOP).with_permutation();
    assert!(!p.is_top());
    assert_eq!(p.at(2, &s).range, Range::new(0, 3));
    assert!(!p.slice(1, 3, &s).is_permutation(), "a part of a permutation is not one");
}

const ALWAYS: FactGuard = FactGuard::Always;

/// The guard of code running where the Boolean values `atoms` all hold.
fn when(atoms: &[u64]) -> FactGuard {
    FactGuard::When(atoms.iter().map(|&a| crate::types::ValueId(a)).collect())
}

#[test]
fn store_finds_facts_through_cells_and_views() {
    let s = NoScalars;
    let mut store = ArrayFacts::new();
    let cells = stmts(&[100, 101, 102, 103]);
    let inc = ElemFacts { range: Range::new(0, 3), order: Some(Order::Increasing), affine: None };
    store.state(cells.clone(), Segmentation::uniform(4, inc), ALWAYS);
    assert_eq!(store.cell_interval(Cell::Stmt(102), &s, &ALWAYS), (Some(0), Some(3)));
    assert_eq!(store.cell_interval(Cell::Stmt(999), &s, &ALWAYS), (None, None));
    assert_eq!(store.cell_interval(Cell::Const(7), &s, &ALWAYS), (Some(7), Some(7)));
    // A slice keeps the order; a reversed view reverses it.
    let view = store.list_facts(&stmts(&[101, 102]), &s, &ALWAYS);
    assert_eq!(view.len(), 1, "a run of the list");
    assert_eq!(view[0].hull(&s).order, Some(Order::Increasing));
    let rev = store.list_facts(&stmts(&[103, 102, 101]), &s, &ALWAYS);
    assert_eq!(rev[0].hull(&s).order, Some(Order::Decreasing));
    // Not a run: no relational fact (element facts still apply per cell).
    assert!(store.list_facts(&stmts(&[100, 102]), &s, &ALWAYS).is_empty());
    assert_eq!(store.hull(&stmts(&[100, 102]), &s, &ALWAYS), Range::new(0, 3));
}

#[test]
fn store_has_no_fact_about_a_list_after_a_write() {
    // AF-1: `a = [c0, c1, c2]` with a fact; after `a[0] = x` the array's
    // cells are `[x, c1, c2]`, and nothing about `x` was stated.
    let s = NoScalars;
    let mut store = ArrayFacts::new();
    store.state(stmts(&[1, 2, 3]), Segmentation::uniform(3, ElemFacts::range(Range::new(0, 4))), ALWAYS);
    let written = stmts(&[50, 2, 3]);
    assert!(store.hull(&written, &s, &ALWAYS).is_top());
    assert!(store.list_facts(&written, &s, &ALWAYS).is_empty());
}

#[test]
fn store_ignores_lists_that_state_nothing() {
    let mut store = ArrayFacts::new();
    assert!(store.state(stmts(&[1, 2]), Segmentation::uniform(2, ElemFacts::TOP), ALWAYS).is_none());
    assert!(store.state(vec![Cell::Const(1)], Segmentation::uniform(1, ElemFacts::range(Range::point(1))), ALWAYS).is_none());
    assert!(store.state(stmts(&[1]), Segmentation::uniform(1, ElemFacts::range(Range::point(1))), FactGuard::Never).is_none());
    assert!(store.cell_facts(Cell::Stmt(1), &NoScalars, &ALWAYS).is_empty());
}

#[test]
fn a_guarded_fact_is_seen_only_where_its_atoms_hold() {
    // Stated where `c1` holds (an `if c1:` arm): seen there and in code that
    // runs under `c1` and more; not on every path, nor under another
    // condition.
    let s = NoScalars;
    let mut store = ArrayFacts::new();
    store.state(stmts(&[5, 6]), Segmentation::uniform(2, ElemFacts::range(Range::new(0, 1))), when(&[1]));
    assert_eq!(store.cell_interval(Cell::Stmt(5), &s, &when(&[1])), (Some(0), Some(1)));
    assert_eq!(store.cell_interval(Cell::Stmt(5), &s, &when(&[1, 2])), (Some(0), Some(1)));
    assert_eq!(store.cell_interval(Cell::Stmt(5), &s, &ALWAYS), (None, None));
    assert_eq!(store.cell_interval(Cell::Stmt(5), &s, &when(&[2])), (None, None));
    assert_eq!(store.cell_interval(Cell::Stmt(5), &s, &FactGuard::Never), (None, None));
    assert!(store.list_facts(&stmts(&[5, 6]), &s, &ALWAYS).is_empty());
    // A fact for every path is seen everywhere.
    assert!(holds_in(&ALWAYS, &when(&[3])));
}

#[test]
fn constants_carry_their_own_facts() {
    let s = NoScalars;
    let ascending = constant_facts(&[Cell::Const(2), Cell::Const(4), Cell::Const(6)]).expect("constants");
    let facts = ascending.hull(&s);
    assert_eq!(facts.range, Range::new(2, 6));
    assert_eq!(facts.order, Some(Order::Increasing));
    assert_eq!(facts.affine, Some(Affine { base: 2, step: 2 }));
    let ties = constant_facts(&[Cell::Const(1), Cell::Const(1), Cell::Const(3)]).expect("constants");
    assert_eq!(ties.hull(&s).order, Some(Order::NonDecreasing));
    assert_eq!(ties.hull(&s).affine, None);
    let unordered = constant_facts(&[Cell::Const(3), Cell::Const(1), Cell::Const(2)]).expect("constants");
    assert_eq!(unordered.hull(&s).order, None);
    // Floats: order only, never strict (the fixed-point grid may tie them).
    let floats = constant_facts(&[Cell::FloatConst(0.5f64.to_bits()), Cell::FloatConst(1.5f64.to_bits())]).expect("floats");
    assert_eq!(floats.hull(&s).order, Some(Order::NonDecreasing));
    assert!(floats.hull(&s).range.is_top());
    assert!(constant_facts(&[Cell::Const(1), Cell::Stmt(9)]).is_none());
}

#[test]
fn a_weak_write_joins_the_value_into_the_segments_it_may_reach() {
    let s = NoScalars;
    let inc = ElemFacts { range: Range::new(0, 9), order: Some(Order::Increasing), affine: None };
    let seg = Segmentation::uniform(10, inc).with_permutation();
    // Anywhere: every cell is its old value or the written one.
    let anywhere = seg.weak_write(&Range::new(20, 30), None, &s);
    assert_eq!(anywhere.hull(&s).range, Range::new(0, 30));
    assert_eq!(anywhere.hull(&s).order, None, "the write may break the order");
    assert!(!anywhere.is_permutation(), "the write may break the permutation");
    // Live/padding: a write that can reach only the padding keeps the live
    // segment's facts.
    let st = known(&[(7, (3, 3))]);
    let live = ElemFacts { range: Range::new(0, 9), order: Some(Order::Increasing), affine: None };
    let split = Segmentation::live_then_padding(6, Bound::stmt(7), live, ElemFacts::range(Range::point(0)));
    let padded = split.weak_write(&Range::point(5), Some((4, 5)), &st);
    assert_eq!(padded.at(0, &st), live);
    assert_eq!(padded.at(5, &st).range, Range::new(0, 5));
}

#[test]
fn a_strong_write_splits_its_segment() {
    let s = NoScalars;
    let inc = ElemFacts { range: Range::new(0, 9), order: Some(Order::Increasing), affine: None };
    let seg = Segmentation::uniform(10, inc);
    let written = seg.strong_write(4, &Range::point(100), &s);
    assert_eq!(written.at(4, &s).range, Range::point(100));
    assert_eq!(written.at(3, &s), inc, "the cells before keep their facts");
    assert_eq!(written.at(5, &s), inc, "the cells after keep their facts");
    assert_eq!(written.hull(&s).range, Range::new(0, 100));
    assert_eq!(written.hull(&s).order, None);
    // A permutation's range survives on the cells the write does not touch.
    let perm = Segmentation::uniform(4, ElemFacts::TOP).with_permutation();
    let p = perm.strong_write(0, &Range::point(-1), &s);
    assert!(!p.is_permutation());
    assert_eq!(p.at(2, &s).range, Range::new(0, 3));
    assert_eq!(p.at(0, &s).range, Range::point(-1));
}

#[test]
fn a_merge_keeps_what_both_sides_have() {
    let s = NoScalars;
    let a = Segmentation::uniform(3, ElemFacts { range: Range::new(0, 2), order: Some(Order::Increasing), affine: None });
    let b = Segmentation::uniform(3, ElemFacts { range: Range::new(5, 9), order: Some(Order::NonDecreasing), affine: None });
    let merged = a.either(&b, &s);
    assert_eq!(merged.hull(&s).range, Range::new(0, 9));
    assert_eq!(merged.hull(&s).order, Some(Order::NonDecreasing), "the array is one side or the other, as a whole");
    let c = Segmentation::uniform(3, ElemFacts::range(Range::new(1, 1)));
    assert_eq!(a.either(&c, &s).hull(&s).order, None);
    let p = Segmentation::uniform(3, ElemFacts::TOP).with_permutation();
    assert!(p.either(&p, &s).is_permutation());
    assert!(!p.either(&a, &s).is_permutation());
}
