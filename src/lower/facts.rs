//! Free facts: facts that hold without any construction, so requirements
//! they imply cost nothing.
//!
//! Two sources, both read off the IR's statements (never off witness
//! values, never off the program's own checks):
//!
//! * **Bounds** from forward interval propagation over the statements that
//!   define a value (`optim::range::RangeResolver`). The provenance table's
//!   intervals are not used: some are deposited by the very checks the
//!   layer may drop (`builder/guard.rs`), and a check must not justify
//!   itself.
//! * **Gadget postconditions**: what the constraints of a scalar statement
//!   the program emits anyway prove about its operands, on every backend.

use std::cell::RefCell;

use crate::ir::IRGraph;
use crate::optim::{RangeResolver, Resolver};
use crate::types::{ScalarValue, StmtId, Value};

use super::graph::{Bound, Claim, Fact, ListEntry};

pub struct FreeFacts<'a> {
    graph: &'a IRGraph,
    range: RefCell<RangeResolver>,
    /// Facts established by the postconditions of scalar gadgets.
    established: Vec<Fact>,
    /// Prove nothing (the ablation `ZINNIA_ABLATE_LAYER_FACTS`).
    disabled: bool,
}

impl<'a> FreeFacts<'a> {
    pub fn new(graph: &'a IRGraph) -> FreeFacts<'a> {
        FreeFacts { graph, range: RefCell::new(RangeResolver::new()), established: Vec::new(), disabled: false }
    }

    /// No free fact at all (the ablation `ZINNIA_ABLATE_LAYER_FACTS`).
    pub fn none(graph: &'a IRGraph) -> FreeFacts<'a> {
        FreeFacts { disabled: true, ..FreeFacts::new(graph) }
    }

    /// `true` when `fact` holds without any construction.
    pub fn proves(&self, fact: &Fact) -> bool {
        !self.disabled && (self.established_by_gadget(fact) || self.proves_by_bounds(&fact.claim))
    }

    /// `true` when a scalar gadget's postcondition implies `fact`, or it
    /// holds by the definition of its statements (`array_facts::
    /// definitional`: a sort's order, a square's sign). Program checks are
    /// discharged only this way: a check that bounds alone prove is the job
    /// of `AlwaysSatisfiedElimination`, which runs earlier with the same
    /// range engine and SMT.
    pub fn established_by_gadget(&self, fact: &Fact) -> bool {
        !self.disabled
            && (self.established.iter().any(|f| f.implies(fact))
                || super::array_facts::definitional(self.graph, &fact.claim))
    }

    fn proves_by_bounds(&self, claim: &Claim) -> bool {
        match *claim {
            Claim::InRange { x, lo, hi } => {
                let Some((min, max)) = self.bounds(x) else { return false };
                let hi_min = match hi {
                    Bound::Const(h) => Some(h),
                    Bound::Stmt(h) => self.bounds(h).map(|(m, _)| m),
                };
                lo <= min && hi_min.is_some_and(|h| max < h)
            }
            Claim::NonNegative(x) => self.bounds(x).is_some_and(|(min, _)| min >= 0),
            Claim::NonZero(x) => self.bounds(x).is_some_and(|(min, max)| min > 0 || max < 0),
            Claim::Equal(a, b) => a == b,
            Claim::Permutation(a, b) => a == b,
            Claim::Within { x, lo, hi } => self.within(x, lo, hi),
            Claim::Ordered { x, y, strict } => {
                x == y && !strict
                    || matches!((self.max(x), self.min(y)), (Some(a), Some(b)) if if strict { a < b } else { a <= b })
            }
            _ if !super::array_facts::enabled() => false,
            Claim::ElementsWithin { list, lo, hi } => list.entries().iter().all(|entry| match *entry {
                ListEntry::Stmt(x) => self.within(x, lo, hi),
                ListEntry::Int(c) => lo.is_none_or(|l| l <= c) && hi.is_none_or(|u| c <= u),
            }),
            Claim::NonDecreasing(list) | Claim::StrictlyIncreasing(list) => {
                let strict = matches!(claim, Claim::StrictlyIncreasing(_));
                list.entries().windows(2).all(|pair| {
                    let (a, b) = (self.entry_bounds(pair[0]), self.entry_bounds(pair[1]));
                    matches!((a.1, b.0), (Some(a), Some(b)) if if strict { a < b } else { a <= b })
                })
            }
            Claim::CountBound { list, max_nonzero } => list.len() <= max_nonzero as usize,
            Claim::Distinct(_) => false,
        }
    }

    /// Whether the bounds of `x` lie in `[lo, hi]` (an open end bounds
    /// nothing; each end is resolved on its own).
    fn within(&self, x: StmtId, lo: Option<i64>, hi: Option<i64>) -> bool {
        lo.is_none_or(|l| self.min(x).is_some_and(|m| l <= m)) && hi.is_none_or(|u| self.max(x).is_some_and(|m| m <= u))
    }

    fn entry_bounds(&self, entry: ListEntry) -> (Option<i64>, Option<i64>) {
        match entry {
            ListEntry::Int(c) => (Some(c), Some(c)),
            ListEntry::Stmt(s) => (self.min(s), self.max(s)),
        }
    }

    /// The least value of an integer statement, when known.
    fn min(&self, stmt: StmtId) -> Option<i64> {
        if self.disabled {
            return None;
        }
        let value = Value::Integer(ScalarValue::new(None, Some(stmt)));
        self.range.borrow_mut().resolve_min_with_stmts(&value, &self.graph.stmts)
    }

    /// The greatest value of an integer statement, when known.
    fn max(&self, stmt: StmtId) -> Option<i64> {
        if self.disabled {
            return None;
        }
        let value = Value::Integer(ScalarValue::new(None, Some(stmt)));
        self.range.borrow_mut().resolve_max_with_stmts(&value, &self.graph.stmts)
    }

    /// Bounds of an integer statement's value, when known.
    pub fn bounds(&self, stmt: StmtId) -> Option<(i64, i64)> {
        if self.disabled {
            return None;
        }
        let value = Value::Integer(ScalarValue::new(None, Some(stmt)));
        let mut range = self.range.borrow_mut();
        let min = range.resolve_min_with_stmts(&value, &self.graph.stmts)?;
        let max = range.resolve_max_with_stmts(&value, &self.graph.stmts)?;
        Some((min, max))
    }
}
