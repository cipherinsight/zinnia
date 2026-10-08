//! Array facts the program states about its inputs (`docs/array-facts.md`):
//! the element predicates of `@requires`, and Boolean input arrays.
//!
//! Each fact is stated on the input's cells (the statements reading its
//! elements), so it describes the input as passed and says nothing about the
//! array after a write (a written array has other cells).

use crate::optim::array_facts::{Bound, ElemFacts, Order, Range, Segmentation};
use crate::types::Value;

use super::IRGenerator;

impl IRGenerator {
    /// State what a structural `@requires` predicate enforces on every
    /// element of its input array. Called right after the predicate's
    /// witness emitter has asserted it for every element (under the current
    /// execution predicate, at entry), which is what makes the fact hold
    /// wherever that predicate does:
    ///
    /// * `forall(a, in_range(lo, hi))`: `lo <= a[i] <= hi`;
    /// * `forall(a, lt_bound(B))`: `a[i] <= B - 1`;
    /// * `forall(a, is_nonneg)`: `a[i] >= 0`;
    /// * `forall(a, eq_const(C))`: `a[i] == C`;
    /// * `is_sorted(a)`: `a[i] <= a[i + 1]` in row-major order.
    ///
    /// Bounds are integer literals or integer scalars (a symbolic bound);
    /// anything else (a float bound, compared in floating point) states
    /// nothing, and so does a float array (its range would be about the
    /// fixed-point encoding).
    pub(crate) fn state_requires_array_facts(&mut self, kind: &str, args: &[String]) {
        let Some(array) = args.first().and_then(|name| self.ctx.get(name)) else { return };
        if matches!(kind, "is_sorted" | "is_monotone_nondecreasing") {
            // `is_sorted(a)`: every consecutive pair, row-major, is asserted
            // `a[i] <= a[i + 1]`.
            let Some(cells) = self.builder.leaf_cells(&array) else { return };
            let order = ElemFacts { order: Some(Order::NonDecreasing), ..ElemFacts::TOP };
            let guard = self.builder.facts.guard().clone();
            self.builder.state_array_facts(&cells, Segmentation::uniform(cells.len(), order), guard);
            return;
        }
        let bound = |g: &Self, k: usize| args.get(k).and_then(|name| g.integer_bound(name));
        let range = match kind {
            "forall_in_range" => Range { lo: bound(self, 1), hi: bound(self, 2) },
            "forall_lt_bound" => Range { lo: None, hi: bound(self, 1).and_then(|b| b.shifted(-1)) },
            "forall_is_nonneg" => Range::at_least(0),
            "forall_eq_const" => Range { lo: bound(self, 1), hi: bound(self, 1) },
            _ => return,
        };
        if range.is_top() {
            return;
        }
        let Some(cells) = self.builder.leaf_cells(&array) else { return };
        let len = cells.len();
        let guard = self.builder.facts.guard().clone();
        self.builder.state_array_facts(&cells, Segmentation::uniform(len, ElemFacts::range(range)), guard);
    }

    /// A `@requires` argument as an integer bound: a literal, or an integer
    /// scalar (a constant, or the statement computing it).
    fn integer_bound(&self, name: &str) -> Option<Bound> {
        match self.ctx.get(name) {
            Some(v @ Value::Integer(_)) => match (v.int_val(), v.stmt_id()) {
                (Some(c), _) => Some(Bound::Const(c)),
                (None, Some(stmt)) => Some(Bound::stmt(stmt)),
                (None, None) => None,
            },
            Some(_) => None,
            None => name.parse::<i64>().ok().map(Bound::Const),
        }
    }

    /// State that a Boolean input array's elements are 0 or 1: each was read
    /// with its domain check `x * (x - 1) == 0` (`input_domain`), emitted on
    /// every path when the input is read.
    pub(crate) fn state_boolean_input_facts(&mut self, array: &Value) {
        let Some(cells) = self.builder.leaf_cells(array) else { return };
        let len = cells.len();
        self.builder.state_defined_facts(&cells, Segmentation::uniform(len, ElemFacts::range(Range::boolean())));
    }
}
