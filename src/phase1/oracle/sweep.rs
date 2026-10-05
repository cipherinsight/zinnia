//! The CANFAIL sweep of `revision/phase1/README.md` §4 step 1, as one
//! incremental solver with a subsumption ledger in front of it.
//!
//! Check k is asked over Φ_{position(k)−1}: the base of Φ (ranges,
//! preconditions, declarations, product facts) and the checks before k. Asked
//! through the general path — one sliced solver per (epoch, cone) — the
//! sweep is quadratic on an unrolled loop: every check mentions the loop's
//! one shape symbol, so the cone of check k is the k−1 checks before it, and
//! nothing is ever reused because every check is a new epoch. Two
//! consequences of the definition remove that:
//!
//! 1. **One solver, in check order.** Φ_{k} is Φ_{k−1} plus check k, so the
//!    sweep is a single solver that asserts the base once and then, after
//!    check k has been asked, asserts check k itself. Each constraint is
//!    asserted exactly once and Z3 keeps what it learned; the question for
//!    check k is a `push`, the assertion of φ_k ∧ ¬defcond_k, a `check` and a
//!    `pop`.
//! 2. **Subsumption.** An unrolled loop asks `i < n` for i = 0, 1, 2, …; once
//!    `j < n` has been recorded, every later `i < n` with i ≤ j under a path
//!    condition no stronger than j's is implied by Φ_{position−1} and needs
//!    no solver. The [`Ledger`] keeps, per term and side, the bounds the
//!    checks so far have placed on it, and per defcond conjunct the path
//!    conditions it was checked under.
//!
//! The answers land in A like every other question; the layer is
//! `Syntactic` for a subsumed check and `Solver` for one the sweep solver
//! answered.

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use z3::{SatResult, Solver};

use crate::phase0::phi::ConstraintKind;
use crate::phase0::term::{BoolId, NodeView, TermArena};

use super::answers::{Answer, AnswerId, Evidence, Layer, Verdict};
use super::layers::{self, Trivalent};
use super::questions::{Question, QuestionKey};
use super::{outcome_evidence, Oracle};

/// One bound a check placed on a term: `term ≥ value` (`lower`) or
/// `term ≤ value`, under `path`, recorded at check `position`.
#[derive(Debug, Clone)]
struct Bound {
    value: i128,
    path: BoolId,
    position: u32,
}

/// What the checks so far imply, in the two shapes the sweep recognises.
#[derive(Debug, Default)]
pub struct Ledger {
    /// (term index, is a lower bound) → the bounds recorded on it.
    bounds: HashMap<(u32, bool), Vec<Bound>>,
    /// A defcond conjunct → the (path, position) pairs it was checked under.
    repeats: HashMap<u32, Vec<(BoolId, u32)>>,
}

impl Ledger {
    /// Record check `position`: `path ⇒ defcond`, conjunct by conjunct.
    fn add(&mut self, terms: &TermArena, path: BoolId, defcond: BoolId, position: u32) {
        for conjunct in layers::conjuncts(terms, defcond) {
            self.repeats.entry(conjunct).or_default().push((path, position));
            if let Some((term, lower, value)) = bound_of(terms, conjunct) {
                self.bounds
                    .entry((term, lower))
                    .or_default()
                    .push(Bound { value, path, position });
            }
        }
    }

    /// The position of an earlier check that implies `conjunct` under a path
    /// condition no stronger than `path_conjuncts`, if there is one.
    fn implies(
        &self,
        terms: &TermArena,
        conjunct: u32,
        path_conjuncts: &BTreeSet<u32>,
    ) -> Option<(u32, String)> {
        let no_stronger =
            |path: BoolId| layers::conjuncts(terms, path).is_subset(path_conjuncts);
        if let Some(seen) = self.repeats.get(&conjunct) {
            if let Some((_, position)) = seen.iter().find(|(path, _)| no_stronger(*path)) {
                return Some((*position, "the same condition".to_string()));
            }
        }
        let (term, lower, value) = bound_of(terms, conjunct)?;
        let bounds = self.bounds.get(&(term, lower))?;
        let stronger = bounds.iter().find(|bound| {
            let at_least = if lower { bound.value >= value } else { bound.value <= value };
            at_least && no_stronger(bound.path)
        })?;
        let sign = if lower { "≥" } else { "≤" };
        let shown = terms.int_at(term).map(|t| terms.pretty(t)).unwrap_or_default();
        Some((stronger.position, format!("the bound {shown} {sign} {}", stronger.value)))
    }
}

/// `t ≥ c` or `t ≤ c` when `conjunct` is a comparison with a literal on one
/// side: (term index, is a lower bound, c).
fn bound_of(terms: &TermArena, conjunct: u32) -> Option<(u32, bool, i128)> {
    let (a, b, strict) = match terms.node_at(conjunct as usize) {
        NodeView::Lt(a, b) => (a, b, true),
        NodeView::Le(a, b) => (a, b, false),
        _ => return None,
    };
    match (terms.node_at(a as usize), terms.node_at(b as usize)) {
        // c < t  ⇔  t ≥ c + 1;  c ≤ t  ⇔  t ≥ c
        (NodeView::Const(c), _) => Some((b, true, if strict { c.checked_add(1)? } else { c })),
        // t < c  ⇔  t ≤ c − 1;  t ≤ c
        (_, NodeView::Const(c)) => Some((a, false, if strict { c.checked_sub(1)? } else { c })),
        _ => None,
    }
}

/// The sweep's solver: the base of Φ, then the checks in order.
pub(super) struct SweepSolver {
    solver: Solver,
    /// Constraints with index below this have been asserted (checks
    /// included) or are checks still to come.
    asserted_below: usize,
}

impl<'a> Oracle<'a> {
    /// §4 step 1 for every check of Φ, in order: (constraint index, answer).
    pub fn canfail_sweep(&mut self, terms: &TermArena) -> Vec<(usize, AnswerId)> {
        let mut out = Vec::new();
        let mut ledger = Ledger::default();
        let mut position = 0;
        let always = terms.tt();
        for at in 0..self.phi.constraints().len() {
            let (path, defcond) = match &self.phi.constraints()[at].kind {
                ConstraintKind::Check { path, defcond, .. } => (*path, *defcond),
                _ => continue,
            };
            position += 1;
            let key = QuestionKey {
                question: Question::CanFail { path, defcond },
                epoch: position - 1,
                condition: always,
            };
            self.stats.kind_mut(key.question.kind()).asked += 1;
            let answer = if let Some(existing) = self.answers.lookup(&key) {
                self.stats.kind_mut(key.question.kind()).memo_hits += 1;
                existing
            } else {
                let answer = self
                    .syntactic(terms, &key)
                    .or_else(|| self.interval_layer(terms, &key))
                    .or_else(|| self.subsumed(terms, &key, &ledger, path, defcond))
                    .unwrap_or_else(|| self.sweep_solver_layer(terms, &key, at, path, defcond));
                match answer.layer {
                    Layer::Syntactic => self.stats.kind_mut(key.question.kind()).syntactic += 1,
                    Layer::Interval => self.stats.kind_mut(key.question.kind()).interval += 1,
                    Layer::Solver => self.stats.kind_mut(key.question.kind()).solver += 1,
                }
                self.answers.record(answer)
            };
            self.answers.mark_root(answer);
            if self.answers.get(answer).map(|a| a.verdict == Verdict::CanFail(false)) == Some(true) {
                self.redundant.insert(at);
            }
            ledger.add(terms, path, defcond, position);
            out.push((at, answer));
        }
        out
    }

    /// The ledger's answer: every conjunct of the defcond is either entailed
    /// by the ranges or implied by an earlier check.
    fn subsumed(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        ledger: &Ledger,
        path: BoolId,
        defcond: BoolId,
    ) -> Option<Answer> {
        let path_conjuncts = layers::conjuncts(terms, path);
        let mut why = Vec::new();
        for conjunct in layers::conjuncts(terms, defcond) {
            let as_bool = terms.bool_at(conjunct)?;
            if layers::evaluate_bool(terms, self.phi, as_bool) == Trivalent::True {
                continue;
            }
            let (position, what) = ledger.implies(terms, conjunct, &path_conjuncts)?;
            why.push(format!("{what} from check {position}"));
        }
        Some(Answer {
            key: *key,
            layer: Layer::Syntactic,
            verdict: Verdict::CanFail(false),
            evidence: Evidence::Syntactic(format!(
                "implied by the checks before it: {}",
                why.join("; ")
            )),
            tight: true,
            millis: 0,
            cone_constraints: 0,
            cone_variables: 0,
            interval_bound: None,
        })
    }

    /// Ask the sweep solver, with every check before `at` asserted.
    fn sweep_solver_layer(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        at: usize,
        path: BoolId,
        defcond: BoolId,
    ) -> Answer {
        let asserted = self.sweep_catch_up(terms, at);
        if self.spent_millis >= self.limits.program_millis {
            return Answer {
                key: *key,
                layer: Layer::Solver,
                verdict: Verdict::CanFail(true),
                evidence: Evidence::Unknown("the program's solver budget is spent".to_string()),
                tight: true,
                millis: 0,
                cone_constraints: asserted,
                cone_variables: 0,
                interval_bound: None,
            };
        }
        let taken = self.z3.boolean(terms, path);
        let failing = self.z3.boolean(terms, defcond).not();
        let started = Instant::now();
        let (result, evidence) = {
            let solver = &self.sweep.as_ref().expect("built by catch_up").solver;
            solver.push();
            solver.assert(&taken);
            solver.assert(&failing);
            let result = solver.check();
            let evidence = outcome_evidence(solver, result);
            solver.pop(1);
            (result, evidence)
        };
        let millis = started.elapsed().as_millis();
        self.spent_millis += millis;
        self.stats.solver.millis += millis;
        self.stats.solver.sweep_queries += 1;
        self.stats.solver.sweep_millis += millis;
        self.charge(result);
        Answer {
            key: *key,
            layer: Layer::Solver,
            verdict: Verdict::CanFail(result != SatResult::Unsat),
            evidence,
            tight: true,
            millis,
            cone_constraints: asserted,
            cone_variables: 0,
            interval_bound: None,
        }
    }

    /// Bring the sweep solver up to Φ_{position(at)−1}: build it on first use
    /// with the base of Φ, then assert every check with index below `at` not
    /// asserted yet. Returns how many constraints it holds.
    fn sweep_catch_up(&mut self, terms: &TermArena, at: usize) -> usize {
        if self.sweep.is_none() {
            let solver = Solver::new();
            solver.set_params(&self.params());
            let base: Vec<usize> = (0..self.phi.constraints().len())
                .filter(|&index| self.layers.position(index).is_none())
                .collect();
            let products: Vec<usize> = (0..self.phi.products().len()).collect();
            let cone = layers::Cone { constraints: base, products, variables: 0 };
            for assertion in self.assertions(terms, &cone) {
                solver.assert(&assertion);
            }
            self.stats.solver.sweep_asserted += cone.constraints.len() as u64;
            self.stats.solver.solvers_built += 1;
            self.sweep = Some(SweepSolver { solver, asserted_below: 0 });
        }
        let from = self.sweep.as_ref().expect("just built").asserted_below;
        for index in from..at {
            if self.layers.position(index).is_some() {
                let formula = self.phi.constraints()[index].formula;
                let assertion = self.z3.boolean(terms, formula);
                self.sweep.as_ref().expect("just built").solver.assert(&assertion);
                self.stats.solver.sweep_asserted += 1;
            }
        }
        let sweep = self.sweep.as_mut().expect("just built");
        sweep.asserted_below = sweep.asserted_below.max(at);
        self.stats.solver.sweep_asserted as usize
    }
}
