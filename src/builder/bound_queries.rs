//! Bound queries of the IR generator, cheapest tier first
//! (`docs/array-facts.md`).
//!
//! Two kinds of question are asked about an integer value `v`:
//!
//! * a **bound query** (`IRBuilder::ask_bounds`): "what are `min(v)` and
//!   `max(v)`?". A sizing site (a loop's trip count, a capacity) needs the
//!   number;
//! * a **decision query** ([`IRBuilder::proves_at_least`],
//!   [`IRBuilder::proves_at_most`], [`IRBuilder::proves_within`]): "does
//!   `v >= c` hold wherever the current code runs?". A decision site (skip a
//!   wrap, drop a check, pick the Boolean form of `&`) needs only yes or
//!   don't-know.
//!
//! A decision tries the tiers that use no solver first
//! ([`IRBuilder::cheap_bounds`]: constants, intervals over the IR and the
//! scalar facts, then the array facts of the arrays `v` is read from), then
//! one satisfiability check of the violation (`v < c`, through
//! `Resolver::proves_bound_with_stmts`). That is never weaker than reading
//! the extreme of a bound query (if `min(v) = m >= c` is provable, so is
//! `v >= c`), and much cheaper when the answer is no: Z3 finds a
//! counter-model instead of searching for an optimum until its step budget
//! runs out. "Don't know" is never read as "no bound holds".

use crate::optim::array_facts::Cell;
use crate::optim::resolver::{Effort, Side};
use crate::optim::Resolver;
use crate::types::Value;

use super::array_facts::Where;
use super::IRBuilder;

impl IRBuilder {
    /// Bounds of an integer value from the tiers that use no solver, in
    /// context `at`: its constant, interval propagation over its definition,
    /// the scalar facts on it (and one step of arithmetic over facts), and
    /// the array facts of the arrays it is a cell of or is read from. Each
    /// end is the tightest any tier gives; `None` where none does.
    pub fn cheap_bounds(&self, v: &Value, at: Where) -> (Option<i64>, Option<i64>) {
        let (mut lo, mut hi) = self.scalar_bounds(v, at);
        if let Some(cell) = Cell::of(v).filter(|_| !matches!(v, Value::Float(_))) {
            let (min, max) = self.array_fact_range(&cell, v, at).interval(&self.oracle(at));
            lo = tighter(lo, min, i64::max);
            hi = tighter(hi, max, i64::min);
        }
        (lo, hi)
    }

    /// [`Self::cheap_bounds`] without the array facts: the oracle that
    /// array facts use for their own symbolic bounds, so a lookup never
    /// recurses into the store. In context `Everywhere` only what holds on
    /// every path is used: constants, intervals over the IR (statements mean
    /// the same on every path) and the facts that hold on every path.
    pub(crate) fn scalar_bounds(&self, v: &Value, at: Where) -> (Option<i64>, Option<i64>) {
        if let Some(n) = v.int_val() {
            return (Some(n), Some(n));
        }
        let (mut lo, mut hi) = {
            let mut range = self.array_facts.intervals.lock().expect("array facts");
            (range.resolve_min_with_stmts(v, &self.stmts), range.resolve_max_with_stmts(v, &self.stmts))
        };
        let mut take = |b: Option<(i64, i64)>| {
            if let Some((min, max)) = b {
                lo = tighter(lo, Some(min), i64::max);
                hi = tighter(hi, Some(max), i64::min);
            }
        };
        let Some(vid) = v.value_id() else { return (lo, hi) };
        match at {
            Where::Everywhere => take(crate::optim::resolver::derive_bounds_from_unconditional_facts(&self.facts, vid)),
            Where::Here => {
                take(crate::optim::resolver::derive_bounds_from_facts(&self.facts, vid));
                take(self.scalar_value_bounds.get(&vid).copied());
                take(self.ask_bounds_via_arith(v));
                take(self.deep_bounds_walk(v));
            }
        }
        (lo, hi)
    }

    /// `v >= bound` wherever the current code runs: the cheap tiers, then
    /// one satisfiability check. `false` means not proved.
    pub fn proves_at_least(&mut self, v: &Value, bound: i64) -> bool {
        self.proves_side(v, bound, Side::AtLeast)
    }

    /// `v <= bound` wherever the current code runs (see
    /// [`Self::proves_at_least`]).
    pub fn proves_at_most(&mut self, v: &Value, bound: i64) -> bool {
        self.proves_side(v, bound, Side::AtMost)
    }

    /// `lo <= v <= hi` wherever the current code runs (both ends inclusive).
    pub fn proves_within(&mut self, v: &Value, lo: i64, hi: i64) -> bool {
        self.proves_at_least(v, lo) && self.proves_at_most(v, hi)
    }

    fn proves_side(&mut self, v: &Value, bound: i64, side: Side) -> bool {
        let (lo, hi) = self.cheap_bounds(v, Where::Here);
        let end = match side {
            Side::AtLeast => lo,
            Side::AtMost => hi,
        };
        let proved = match side {
            Side::AtLeast => end.is_some_and(|l| l >= bound),
            Side::AtMost => end.is_some_and(|h| h <= bound),
        };
        if proved {
            return true;
        }
        // An end the cheap tiers give but not tightly enough makes the solver
        // check a speculative tightening, with a smaller budget.
        let effort = if end.is_some() { Effort::Tighten } else { Effort::Full };
        let (resolver, stmts) = self.split_resolver_and_stmts();
        resolver.proves_bound_with_stmts(v, stmts, bound, side, effort)
    }
}

/// The tighter of two optional ends (`pick` is `max` for a lower end, `min`
/// for an upper one).
fn tighter(a: Option<i64>, b: Option<i64>, pick: fn(i64, i64) -> i64) -> Option<i64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(pick(x, y)),
        (x, y) => x.or(y),
    }
}
