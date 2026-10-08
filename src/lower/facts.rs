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
//! * **Operator semantics** of array facts (`docs/array-facts.md` 2.7c): the
//!   argsort lanes of one sorted line, all ranks once, are a permutation of
//!   `0..n` ([`FreeFacts::is_argsort_line`]).

use std::cell::RefCell;

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::optim::{RangeResolver, Resolver};
use crate::types::{ScalarValue, StmtId, Value};

use super::graph::{Bound, Claim, Fact, ListEntry, ListId};

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

    /// `true` when a scalar gadget's postcondition implies `fact`. Program
    /// checks are discharged only this way: a check that bounds alone prove
    /// is the job of `AlwaysSatisfiedElimination`, which runs earlier with
    /// the same range engine and SMT.
    pub fn established_by_gadget(&self, fact: &Fact) -> bool {
        !self.disabled && self.established.iter().any(|f| f.implies(fact))
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
            Claim::Permutation(a, b) => {
                a == b
                    || (!crate::optim::array_facts::ablated()
                        && (self.is_argsort_line(a, b) || self.is_argsort_line(b, a)))
            }
            Claim::Within { x, lo, hi } => {
                self.bounds(x).is_some_and(|(min, max)| lo.is_none_or(|l| l <= min) && hi.is_none_or(|u| max <= u))
            }
        }
    }

    /// Whether `values` are the `n` argsort lanes of one sorted line (one
    /// `OpKind::Sort` instance, output `INDICES`, every rank `0..n` once) and
    /// `targets` is `[0, ..., n - 1]`. NumPy's stable argsort of a line of `n`
    /// values is a permutation of `0..n`, and every construction of the sort
    /// family yields it on every accepted witness (`lower::sort`, each
    /// construction's soundness argument), so the claim holds with no
    /// construction of its own.
    fn is_argsort_line(&self, values: ListId, targets: ListId) -> bool {
        let (values, targets) = (values.entries(), targets.entries());
        let n = values.len();
        let is_range = targets.iter().enumerate().all(|(k, t)| *t == ListEntry::Int(k as i64));
        if n == 0 || targets.len() != n || !is_range {
            return false;
        }
        let mut seen = vec![false; n];
        let mut line: Option<&[StmtId]> = None;
        for entry in values.iter() {
            let ListEntry::Stmt(s) = *entry else { return false };
            let Some(stmt) = self.graph.stmts.get(s as usize) else { return false };
            let IR::ArrayOp(op) = &stmt.ir else { return false };
            let rank = match op.position {
                Some(r) if (r as usize) < n => r as usize,
                _ => return false,
            };
            let lane_of_line = op.kind == super::OpKind::Sort
                && op.output == super::sort::INDICES
                && op.shared as usize == n
                && stmt.arguments.len() == n;
            if !lane_of_line || seen[rank] || line.is_some_and(|l| l != &stmt.arguments[..]) {
                return false;
            }
            seen[rank] = true;
            line = Some(&stmt.arguments);
        }
        true
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::FreeFacts;
    use crate::builder::IRBuilder;
    use crate::circuit_input::InputPath;
    use crate::lower::graph::{Claim, Fact, ListId};
    use crate::types::{StmtId, Value};

    /// Two lines of 3 floats and their argsort lanes (rank order).
    fn argsort_lines() -> (crate::ir::IRGraph, Vec<StmtId>, Vec<StmtId>, Vec<StmtId>) {
        let mut b = IRBuilder::new();
        let line = |b: &mut IRBuilder, name: &str| -> Arc<[Value]> {
            (0..3).map(|i| b.ir_read_float(InputPath::new(&format!("{name}{i}"), vec![]), false)).collect()
        };
        let (x, y) = (line(&mut b, "x"), line(&mut b, "y"));
        let lanes = |b: &mut IRBuilder, l: &Arc<[Value]>, index: bool| -> Vec<StmtId> {
            (0..3).map(|r| b.ir_sorted(l, r, index).stmt_id().expect("a statement")).collect()
        };
        let ax = lanes(&mut b, &x, true);
        let ay = lanes(&mut b, &y, true);
        let sx = lanes(&mut b, &x, false);
        (b.export_ir_graph(), ax, ay, sx)
    }

    fn permutation(values: &[StmtId], n: usize) -> Fact {
        Fact::everywhere(Claim::Permutation(ListId::of_stmts(values), ListId::range(n)))
    }

    #[test]
    fn the_argsort_lanes_of_a_line_are_a_permutation() {
        let (graph, ax, ay, sx) = argsort_lines();
        let facts = FreeFacts::new(&graph);
        assert!(facts.proves(&permutation(&ax, 3)));
        assert!(facts.proves(&permutation(&[ax[2], ax[0], ax[1]], 3)), "any order of the ranks");
        assert!(!facts.proves(&permutation(&[ax[0], ax[1]], 3)), "a missing rank");
        assert!(!facts.proves(&permutation(&[ax[0], ax[0], ax[1]], 3)), "a repeated rank");
        assert!(!facts.proves(&permutation(&[ax[0], ax[1], ay[2]], 3)), "lanes of two lines");
        assert!(!facts.proves(&permutation(&sx, 3)), "sorted values are not positions");
        assert!(!FreeFacts::none(&graph).proves(&permutation(&ax, 3)), "the ablation proves nothing");
    }
}
