//! Statements the established facts decide: comparisons whose outcome the
//! ranges of their operands fix, and the maximum or minimum of two values
//! one of which the facts prove the larger ([`Decided`]).
//!
//! This is where element-range facts make `np.clip`, `np.maximum`,
//! `np.minimum`, `np.where(a > c, a, c)` and comparisons the identity or a
//! constant: `np.clip(a, 0, 255)` of an `a` checked into `[0, 255]` is `a`
//! (its `select(a > 0, a, 0)` is `a`: the two differ only where `a < 0`,
//! which the check excludes), and `a < 300` is `true`.
//!
//! **The facts.** Every fact holds on every accepted witness of the lowered
//! program:
//!
//! * a program check the selection keeps (`Within`, `InRange`,
//!   `NonNegative` and `Ordered` forms holding everywhere);
//! * a check the layer emits (a merged range check, a precondition's
//!   check);
//! * a bound the range engine derives from the statements' definitions
//!   (`facts::FreeFacts`), or a value's order by definition (two value lanes
//!   of one sort, `array_facts::definitional`).
//!
//! **Soundness.** A replaced statement equals its replacement on every
//! witness that satisfies the facts, and the circuit still rejects every
//! other witness: take a witness that violates some fact, and of the
//! violated facts the one whose subject comes first in statement order. Its
//! check's condition reads that subject (directly, or as `x + c`), whose
//! value depends only on earlier statements, and a replacement among those
//! used only facts about statements earlier still, none violated; so the
//! subject and the check are evaluated as in the original program, and the
//! check fails. The condition of a kept check is never itself replaced
//! (that would let a check justify its own removal); its other statements
//! may be.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::checks::ProgramCheck;
use super::facts::FreeFacts;
use super::graph::{Claim, Fact, Scope};

/// The facts a lowering establishes, for deciding statements.
pub(super) struct Decided<'a> {
    graph: &'a IRGraph,
    free: &'a FreeFacts<'a>,
    /// Bounds the established checks state, per statement (tightest).
    ranges: HashMap<StmtId, (Option<i64>, Option<i64>)>,
    /// `x <= y` (`x < y` when strict) the established checks state.
    ordered: HashSet<(StmtId, StmtId, bool)>,
    /// The conditions of the program checks that are kept, never replaced.
    frozen: HashSet<StmtId>,
    /// Off under `ZINNIA_ABLATE_ARRAY_FACTS`: nothing is replaced.
    enabled: bool,
    /// Each replaced maximum or minimum, to the operand that replaced it:
    /// the two are equal on every accepted witness, so the operand's facts
    /// are the statement's (a clip's `min(max(a, lo), hi)`).
    aliases: RefCell<HashMap<StmtId, StmtId>>,
}

impl<'a> Decided<'a> {
    /// The facts of the kept program checks (those not in `dropped`) and of
    /// the checks the layer emits (`emitted`), with the free facts.
    pub(super) fn new(
        graph: &'a IRGraph,
        free: &'a FreeFacts<'a>,
        checks: &[ProgramCheck],
        dropped: &HashSet<StmtId>,
        emitted: &[Fact],
    ) -> Decided<'a> {
        let mut decided = Decided {
            graph,
            free,
            ranges: HashMap::new(),
            ordered: HashSet::new(),
            frozen: HashSet::new(),
            enabled: super::array_facts::enabled() && free.enabled(),
            aliases: RefCell::new(HashMap::new()),
        };
        if !decided.enabled {
            return decided;
        }
        let kept: Vec<&ProgramCheck> = checks.iter().filter(|c| !dropped.contains(&c.stmt)).collect();
        let facts: Vec<&Fact> = kept.iter().flat_map(|c| &c.forms).chain(emitted).collect();
        decided.ranges = super::checks::stated_ranges(facts.iter().copied());
        for fact in facts.iter().filter(|f| f.scope == Scope::Everywhere) {
            if let Claim::Ordered { x, y, strict } = fact.claim {
                decided.ordered.insert((x, y, strict));
            }
        }
        for check in kept {
            decided.frozen.extend(graph.stmts[check.stmt as usize].arguments.iter().copied());
        }
        decided
    }

    /// The bounds of a statement's value (or code): a constant's, the
    /// established checks', and the definitional ones, each end the tighter.
    fn range(&self, s: StmtId) -> (Option<i64>, Option<i64>) {
        let s = self.resolve(s);
        if let Some(c) = super::array_facts::constant_code(self.graph, s) {
            return (Some(c), Some(c));
        }
        let (lo, hi) = self.ranges.get(&s).copied().unwrap_or((None, None));
        let (free_lo, free_hi) = self.free.bounds_of(s);
        (lo.max(free_lo), match (hi, free_hi) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        })
    }

    /// Whether `x <= y` (`x < y` when `strict`) holds on every accepted
    /// witness.
    /// The statement `s` stands for after replacements.
    fn resolve(&self, mut s: StmtId) -> StmtId {
        let aliases = self.aliases.borrow();
        while let Some(&next) = aliases.get(&s) {
            s = next;
        }
        s
    }

    fn at_most(&self, x: StmtId, y: StmtId, strict: bool) -> bool {
        let (x, y) = (self.resolve(x), self.resolve(y));
        if x == y {
            return !strict;
        }
        if self.ordered.contains(&(x, y, true)) || (!strict && self.ordered.contains(&(x, y, false))) {
            return true;
        }
        if super::array_facts::definitional(self.graph, &Claim::Ordered { x, y, strict }) {
            return true;
        }
        match (self.range(x).1, self.range(y).0) {
            (Some(a), Some(b)) => if strict { a < b } else { a <= b },
            _ => false,
        }
    }

    /// The value that replaces input-graph statement `stmt`, when the facts
    /// decide it: a comparison's constant outcome, or the operand a maximum
    /// or minimum always selects. `lookup` holds the lowered operands.
    pub(super) fn replacement(&self, b: &mut IRBuilder, stmt: StmtId, lookup: &HashMap<StmtId, Value>) -> Option<Value> {
        if !self.enabled || self.frozen.contains(&stmt) {
            return None;
        }
        let s = &self.graph.stmts[stmt as usize];
        if let Some(outcome) = self.comparison_outcome(&s.ir, &s.arguments) {
            return Some(b.ir_constant_bool(outcome));
        }
        let (IR::SelectI | IR::SelectF, &[p, t, f]) = (&s.ir, s.arguments.as_slice()) else { return None };
        let condition = &self.graph.stmts[p as usize];
        let &[u, v] = condition.arguments.as_slice() else { return None };
        // The branches as the compared values: the same statements, or
        // statements this emission already replaced by them (the inner
        // `select(a < 0, 0, a)` of a clip that became `a`).
        let lowered = |s: StmtId| lookup.get(&s).and_then(Value::stmt_id);
        let (t, f) = if (t, f) == (u, v) || (lowered(t), lowered(f)) == (lowered(u), lowered(v)) {
            (u, v)
        } else if (t, f) == (v, u) || (lowered(t), lowered(f)) == (lowered(v), lowered(u)) {
            (v, u)
        } else {
            return None;
        };
        // `p` is `u > v` (or `u >= v`) when `greater`, `u < v` (`u <= v`)
        // otherwise; `select(p, u, v)` is then the maximum or the minimum.
        let greater = match condition.ir {
            IR::GtI | IR::GteI | IR::GtF | IR::GteF => true,
            IR::LtI | IR::LteI | IR::LtF | IR::LteF => false,
            _ => return None,
        };
        let maximum = greater == ((t, f) == (u, v));
        // The maximum is `t` when `f <= t`, `f` when `t <= f` (equal values
        // make both the same); the minimum mirrors it.
        let (t_selected, f_selected) = if maximum {
            (self.at_most(f, t, false), self.at_most(t, f, false))
        } else {
            (self.at_most(t, f, false), self.at_most(f, t, false))
        };
        let chosen = if t_selected {
            t
        } else if f_selected {
            f
        } else {
            return None;
        };
        self.aliases.borrow_mut().insert(stmt, chosen);
        Some(lookup[&chosen].clone())
    }

    /// The outcome of the comparison `ir(args)` when the facts fix it.
    fn comparison_outcome(&self, ir: &IR, args: &[StmtId]) -> Option<bool> {
        let &[x, y] = args else { return None };
        let (lower, upper, strict) = match ir {
            IR::LtI | IR::LtF => (x, y, true),
            IR::LteI | IR::LteF => (x, y, false),
            IR::GtI | IR::GtF => (y, x, true),
            IR::GteI | IR::GteF => (y, x, false),
            _ => return None,
        };
        // `lower < upper` (or `<=`) holds, or its negation `upper <= lower`
        // (`upper < lower`) does.
        if self.at_most(lower, upper, strict) {
            Some(true)
        } else if self.at_most(upper, lower, !strict) {
            Some(false)
        } else {
            None
        }
    }
}
