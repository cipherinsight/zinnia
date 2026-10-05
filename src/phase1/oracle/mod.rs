//! The oracle: the only place phase 1 asks a question about Φ.
//!
//! `revision/README.md` §5.5 fixes five kinds of question and §5.6 fixes how
//! they are answered: cheapest layer first. This module is that, with the
//! three requirements the design has to meet before the solver can be
//! trusted at scale.
//!
//! 1. **Layers.** A question is tried syntactically (a literal `true`, a
//!    conjunct of the context, a constant term), then over intervals
//!    ([`layers`]), and only then by Z3.
//! 2. **Cone slicing.** A Z3 query asserts the constraints of Φ_e in the
//!    dependency cone of the question and nothing else, so the per-question
//!    cost does not grow with the size of Φ outside the cone.
//! 3. **Incrementality.** One solver per connected component of Φ, its
//!    checks behind guard literals, and `push`/`pop` around each question,
//!    which enables the guards of the checks in its epoch's cone. Every
//!    constraint is asserted once per component however many epochs ask, so
//!    an unrolled loop — one epoch per value, every check on one symbol —
//!    costs one solver, not one per epoch. The CANFAIL sweep ([`sweep`]) is
//!    the same idea in check order, with a subsumption ledger in front.
//!
//! Every answer lands in A ([`answers`]) with its evidence, and A is the
//! memo: a question is looked up on (kind, term, epoch, g) before it is
//! asked.
//!
//! The term arena is a parameter of every call rather than a field of the
//! oracle. Phase 1 builds terms of its own into it — a comparison it wants
//! proved, the conjunction a question is asked under (`../README.md` §1.1) —
//! so the construction holds it mutably and lends it here for the length of
//! one question. Nothing in this module writes to it.

pub mod answers;
pub mod arena;
pub mod layers;
pub mod questions;
pub mod stats;
pub mod sweep;

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use z3::{Params, SatResult, Solver};

use crate::phase0::interval::Interval;
use crate::phase0::phi::Phi;
use crate::phase0::term::{BoolId, TermArena, TermId};

use answers::{Answer, AnswerId, Answers, Evidence, Layer, Verdict};
use arena::Z3Arena;
use layers::{Cone, Layers, Trivalent};
use questions::{Condition, Question, QuestionKey};
use stats::Stats;

/// Time limits (the brief §3.2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Milliseconds one Z3 query may take, as `phi_solver.py` sets it.
    pub query_millis: u32,
    /// Milliseconds of Z3 work per program, across every query.
    pub program_millis: u128,
    /// Solvers the LRU keeps.
    pub solver_cache: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            query_millis: 3000,
            program_millis: 60_000,
            solver_cache: 16,
        }
    }
}

/// The key of one cached solver: the cone at the last epoch — the connected
/// component of Φ around the question, every check included.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SolverKey {
    constraints: Vec<usize>,
    products: Vec<usize>,
}

/// A cached solver: the component's base constraints asserted outright, each
/// check asserted as `guard_k ⇒ check_k` so that a question at epoch e
/// enables exactly the checks with position ≤ e in its cone.
struct CachedSolver {
    solver: Solver,
    /// Constraint index of a check → its guard literal.
    guards: HashMap<usize, z3::ast::Bool>,
}

/// The oracle over one phase 0 record.
pub struct Oracle<'a> {
    phi: &'a Phi,
    pub layers: Layers,
    z3: Z3Arena,
    pub answers: Answers,
    pub stats: Stats,
    limits: Limits,
    spent_millis: u128,
    solvers: HashMap<SolverKey, CachedSolver>,
    recent: Vec<SolverKey>,
    /// The CANFAIL sweep's solver, built on first use ([`sweep`]).
    sweep: Option<sweep::SweepSolver>,
    /// Checks the sweep found implied by Φ_{position−1}: Φ_e entails them
    /// wherever they are in it, so a query asserts nothing for them. On an
    /// unrolled loop this is most of the checks.
    redundant: HashSet<usize>,
}

impl<'a> Oracle<'a> {
    pub fn new(terms: &TermArena, phi: &'a Phi, limits: Limits) -> Oracle<'a> {
        let mut stats = Stats::default();
        stats.solver.phi_constraints = phi.constraints().len() as u64;
        stats.solver.phi_variables = phi.vars().len() as u64;
        Oracle {
            layers: Layers::new(terms, phi),
            z3: Z3Arena::new(terms),
            phi,
            answers: Answers::default(),
            stats,
            limits,
            spent_millis: 0,
            solvers: HashMap::new(),
            recent: Vec::new(),
            sweep: None,
            redundant: HashSet::new(),
        }
    }

    /// Ask one question, through the memo and the layers.
    pub fn ask(
        &mut self,
        terms: &TermArena,
        question: Question,
        epoch: u32,
        condition: Condition,
    ) -> AnswerId {
        let key = QuestionKey { question, epoch, condition };
        self.stats.kind_mut(question.kind()).asked += 1;
        if let Some(existing) = self.answers.lookup(&key) {
            self.stats.kind_mut(question.kind()).memo_hits += 1;
            return existing;
        }
        let answer = self
            .syntactic(terms, &key)
            .or_else(|| self.interval_layer(terms, &key))
            .unwrap_or_else(|| self.solver_layer(terms, &key));
        match answer.layer {
            Layer::Syntactic => self.stats.kind_mut(question.kind()).syntactic += 1,
            Layer::Interval => self.stats.kind_mut(question.kind()).interval += 1,
            Layer::Solver => self.stats.kind_mut(question.kind()).solver += 1,
        }
        self.answers.record(answer)
    }

    /// The verdict of a question, asked through [`Oracle::ask`].
    pub fn verdict(
        &mut self,
        terms: &TermArena,
        question: Question,
        epoch: u32,
        condition: Condition,
    ) -> Verdict {
        let at = self.ask(terms, question, epoch, condition);
        self.answers.get(at).expect("just recorded").verdict.clone()
    }

    // ── The layers ───────────────────────────────────────────────────

    /// (a) of the brief §3.2.1: a term that decides itself.
    fn syntactic(&mut self, terms: &TermArena, key: &QuestionKey) -> Option<Answer> {
        let evidence = |why: &str| Evidence::Syntactic(why.to_string());
        let (verdict, why) = match key.question {
            Question::Prove(claim) if terms.is_true(claim) => {
                (Verdict::Proved, "the claim is literally true")
            }
            Question::Prove(claim) if is_conjunct(terms, key.condition, claim) => {
                (Verdict::Proved, "the claim is a conjunct of the condition")
            }
            Question::Prove(claim) if terms.is_false(claim) => {
                (Verdict::NotProved, "the claim is literally false")
            }
            Question::Const(term) => {
                let value = terms.constant_value(term)?;
                (Verdict::Constant(value), "the term is a literal")
            }
            Question::ConstBool(claim) => {
                if terms.is_true(claim) {
                    (Verdict::ConstantBool(true), "the term is literally true")
                } else if terms.is_false(claim) {
                    (Verdict::ConstantBool(false), "the term is literally false")
                } else {
                    return None;
                }
            }
            Question::Max(term) | Question::Min(term) => {
                let value = terms.constant_value(term)?;
                (Verdict::Bound(value), "the term is a literal")
            }
            Question::CanFail { path, defcond } => {
                if terms.is_true(defcond) {
                    (Verdict::CanFail(false), "the defcond is literally true")
                } else if terms.is_false(path) {
                    (Verdict::CanFail(false), "the path condition is literally false")
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        Some(Answer {
            key: *key,
            layer: Layer::Syntactic,
            verdict,
            evidence: evidence(why),
            tight: true,
            millis: 0,
            cone_constraints: 0,
            cone_variables: 0,
            interval_bound: interval_bound(terms, self.phi, key),
        })
    }

    /// (b) of the brief §3.2.1: interval arithmetic over the `var_range`
    /// constraints, which hold in every admitted run.
    fn interval_layer(&mut self, terms: &TermArena, key: &QuestionKey) -> Option<Answer> {
        let hull = interval_bound(terms, self.phi, key);
        let finish = |verdict, evidence, tight| {
            Some(Answer {
                key: *key,
                layer: Layer::Interval,
                verdict,
                evidence,
                tight,
                millis: 0,
                cone_constraints: 0,
                cone_variables: 0,
                interval_bound: hull,
            })
        };
        match key.question {
            Question::Prove(claim) => match layers::evaluate_bool(terms, self.phi, claim) {
                // The ranges alone entail the claim.
                Trivalent::True => finish(
                    Verdict::Proved,
                    Evidence::Syntactic("the ranges entail the claim".to_string()),
                    true,
                ),
                // The ranges refute it, so no Φ_e can prove it: not proving
                // only drops a candidate, never accepts one.
                Trivalent::False => finish(
                    Verdict::NotProved,
                    Evidence::Syntactic("the ranges refute the claim".to_string()),
                    true,
                ),
                Trivalent::Unknown => None,
            },
            Question::Const(term) => {
                let range = layers::interval(terms, self.phi, term);
                if let (Some(lo), Some(hi)) = (range.lo, range.hi) {
                    if lo == hi {
                        return finish(
                            Verdict::Constant(lo),
                            Evidence::Interval { lo: Some(lo), hi: Some(hi) },
                            true,
                        );
                    }
                }
                if self.ranges_decide(terms, key) {
                    return finish(
                        Verdict::NotConstant,
                        Evidence::Interval { lo: range.lo, hi: range.hi },
                        true,
                    );
                }
                None
            }
            Question::ConstBool(claim) => match layers::evaluate_bool(terms, self.phi, claim) {
                Trivalent::True => finish(
                    Verdict::ConstantBool(true),
                    Evidence::Syntactic("the ranges entail the term".to_string()),
                    true,
                ),
                Trivalent::False => finish(
                    Verdict::ConstantBool(false),
                    Evidence::Syntactic("the ranges refute the term".to_string()),
                    true,
                ),
                Trivalent::Unknown if self.ranges_decide(terms, key) => finish(
                    Verdict::NotConstant,
                    Evidence::Syntactic("nothing but the ranges bears on the term".to_string()),
                    true,
                ),
                Trivalent::Unknown => None,
            },
            Question::Max(term) | Question::Min(term) => {
                let range = layers::interval(terms, self.phi, term);
                if !self.interval_is_exact(terms, term, key) {
                    return None;
                }
                let bound = match key.question {
                    Question::Max(_) => range.hi,
                    _ => range.lo,
                }?;
                finish(
                    Verdict::Bound(bound),
                    Evidence::Interval { lo: range.lo, hi: range.hi },
                    true,
                )
            }
            Question::CanFail { path, defcond } => {
                let taken = layers::evaluate_bool(terms, self.phi, path);
                let failing = layers::evaluate_bool(terms, self.phi, defcond).not();
                match taken.and(failing) {
                    Trivalent::False => finish(
                        Verdict::CanFail(false),
                        Evidence::Syntactic("the ranges refute the failing condition".to_string()),
                        true,
                    ),
                    _ => None,
                }
            }
        }
    }

    /// Whether nothing but the `var_range` constraints bears on the
    /// question, so that a CONST the intervals did not decide is answered
    /// "not constant" without the solver.
    ///
    /// The answer is conservative in the direction `../../README.md` §5.5
    /// fixes: an `unknown` CONST is treated as non-constant, and a value
    /// that is constant but not recognised gets a wire instead of a literal
    /// — more gates, never a wrong circuit. What the shortcut gives up is
    /// exactly the case the intervals cannot see, `x mod 2 + (x + 1) mod 2`;
    /// what it saves is a Z3 query on every data scalar of a program, whose
    /// interval is the whole dtype range and whose cone is its own range
    /// constraint.
    fn ranges_decide(&mut self, terms: &TermArena, key: &QuestionKey) -> bool {
        if !terms.is_true(key.condition) {
            return false;
        }
        let seeds = key.question.nodes();
        let cone = self.layers.cone(terms, &seeds, key.epoch);
        self.layers.cone_is_ranges_only(&cone)
    }

    /// When the interval bound is the exact answer of a MAX or a MIN.
    ///
    /// `../README.md` §5.6: "if every variable of `e` and of `g` occurs
    /// once, the interval result is the exact answer". That reading holds
    /// over the ranges, so it needs one more condition here: the cone must
    /// hold nothing but ranges, or a `precondition` or a check could make
    /// the true bound smaller than the interval.
    fn interval_is_exact(&mut self, terms: &TermArena, term: TermId, key: &QuestionKey) -> bool {
        if !terms.is_true(key.condition) {
            return false;
        }
        if !self.layers.occurs_once(terms, term.index() as u32) {
            return false;
        }
        let cone = self.layers.cone(terms, &[term.index() as u32], key.epoch);
        self.layers.cone_is_ranges_only(&cone)
    }

    // ── The solver ───────────────────────────────────────────────────

    /// (c) of the brief §3.2.1, with the cone of §3.2.2 and the solver reuse
    /// of §3.2.3.
    fn solver_layer(&mut self, terms: &TermArena, key: &QuestionKey) -> Answer {
        let mut seeds = key.question.nodes();
        if !terms.is_true(key.condition) {
            seeds.push(key.condition.index() as u32);
        }
        let mut cone = self.layers.cone(terms, &seeds, key.epoch);
        cone.constraints.retain(|at| !self.redundant.contains(at));
        if self.spent_millis >= self.limits.program_millis {
            return self.give_up(terms, key, &cone, "the program's solver budget is spent");
        }
        let hull = interval_bound(terms, self.phi, key);
        let started = Instant::now();
        let outcome = self.run(terms, key, &cone);
        let millis = started.elapsed().as_millis();
        self.spent_millis += millis;
        self.stats.solver.millis += millis;
        self.stats.solver.query_millis.add(millis);
        self.stats.solver.cone_constraints.add(cone.constraints.len() as u128);
        self.stats.solver.cone_variables.add(cone.variables as u128);
        Answer {
            key: *key,
            layer: Layer::Solver,
            verdict: outcome.0,
            evidence: outcome.1,
            tight: outcome.2,
            millis,
            cone_constraints: cone.constraints.len(),
            cone_variables: cone.variables,
            interval_bound: hull,
        }
    }

    /// The conservative answer of `../../README.md` §5.5 when the solver
    /// cannot be used: the interval bound for MAX and MIN, "not proved" for
    /// PROVE, "not constant" for CONST, "can fail" for CANFAIL.
    fn give_up(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        why: &str,
    ) -> Answer {
        let (verdict, tight) = match key.question {
            Question::Prove(_) => (Verdict::NotProved, true),
            Question::Const(_) | Question::ConstBool(_) => (Verdict::NotConstant, true),
            Question::CanFail { .. } => (Verdict::CanFail(true), true),
            Question::Max(term) => (self.interval_fallback(terms, term, true), false),
            Question::Min(term) => (self.interval_fallback(terms, term, false), false),
        };
        Answer {
            key: *key,
            layer: Layer::Solver,
            verdict,
            evidence: Evidence::Unknown(why.to_string()),
            tight,
            millis: 0,
            cone_constraints: cone.constraints.len(),
            cone_variables: cone.variables,
            interval_bound: interval_bound(terms, self.phi, key),
        }
    }

    fn interval_fallback(&mut self, terms: &TermArena, term: TermId, upper: bool) -> Verdict {
        let range: Interval = layers::interval(terms, self.phi, term);
        match if upper { range.hi } else { range.lo } {
            Some(bound) => Verdict::Bound(bound),
            None => Verdict::NoBound,
        }
    }

    fn run(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
    ) -> (Verdict, Evidence, bool) {
        match key.question {
            Question::Prove(claim) => self.run_prove(terms, key, cone, claim),
            Question::Const(term) => self.run_const(terms, key, cone, term),
            Question::ConstBool(claim) => self.run_const_bool(terms, key, cone, claim),
            Question::CanFail { path, defcond } => {
                self.run_canfail(terms, key, cone, path, defcond)
            }
            Question::Max(term) => self.run_bound(terms, key, cone, term, true),
            Question::Min(term) => self.run_bound(terms, key, cone, term, false),
        }
    }

    fn run_prove(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        claim: BoolId,
    ) -> (Verdict, Evidence, bool) {
        let condition = self.condition_assertion(terms, key.condition);
        let goal = self.z3.boolean(terms, claim).not();
        let (result, evidence) = {
            let solver = self.solver_for(terms, key.epoch, cone);
            if let Some(assertion) = &condition {
                solver.assert(assertion);
            }
            solver.assert(&goal);
            let result = solver.check();
            let evidence = outcome_evidence(solver, result);
            solver.pop(1);
            (result, evidence)
        };
        self.charge(result);
        let verdict = match result {
            SatResult::Unsat => Verdict::Proved,
            _ => Verdict::NotProved,
        };
        (verdict, evidence, true)
    }

    /// CONST as `../../README.md` §5.6 states it: take a model value, then
    /// ask whether a second one exists.
    fn run_const(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        term: TermId,
    ) -> (Verdict, Evidence, bool) {
        let condition = self.condition_assertion(terms, key.condition);
        let expression = self.z3.int(terms, term);
        let (first, candidate) = {
            let solver = self.solver_for(terms, key.epoch, cone);
            if let Some(assertion) = &condition {
                solver.assert(assertion);
            }
            let first = solver.check();
            let candidate = solver
                .get_model()
                .and_then(|model| model.eval(&expression, true))
                .and_then(|value| value.to_string().parse::<i128>().ok());
            solver.pop(1);
            (first, candidate)
        };
        self.charge(first);
        if first != SatResult::Sat {
            let why = if first == SatResult::Unsat {
                "the condition has no model, so the term has no value"
            } else {
                "the solver did not decide the term"
            };
            return (Verdict::NotConstant, Evidence::Unknown(why.to_string()), true);
        }
        let Some(candidate) = candidate else {
            return (
                Verdict::NotConstant,
                Evidence::Unknown("the model gave no integer for the term".to_string()),
                true,
            );
        };
        let differs = expression.eq(arena::integer(candidate)).not();
        let (second, evidence) = {
            let solver = self.solver_for(terms, key.epoch, cone);
            if let Some(assertion) = &condition {
                solver.assert(assertion);
            }
            solver.assert(&differs);
            let second = solver.check();
            let evidence = outcome_evidence(solver, second);
            solver.pop(1);
            (second, evidence)
        };
        self.charge(second);
        let verdict = match second {
            SatResult::Unsat => Verdict::Constant(candidate),
            _ => Verdict::NotConstant,
        };
        (verdict, evidence, true)
    }

    /// CONST on a Boolean term: one of the two values has no model, or the
    /// term is not constant.
    fn run_const_bool(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        claim: BoolId,
    ) -> (Verdict, Evidence, bool) {
        let condition = self.condition_assertion(terms, key.condition);
        let expression = self.z3.boolean(terms, claim);
        let mut verdict = Verdict::NotConstant;
        let mut evidence = Evidence::Unknown("both values have a model".to_string());
        for value in [true, false] {
            let goal = if value { expression.clone() } else { expression.not() };
            let result = {
                let solver = self.solver_for(terms, key.epoch, cone);
                if let Some(assertion) = &condition {
                    solver.assert(assertion);
                }
                solver.assert(&goal);
                let result = solver.check();
                solver.pop(1);
                result
            };
            self.charge(result);
            if result == SatResult::Unsat {
                // The other value is the only one the term can take.
                verdict = Verdict::ConstantBool(!value);
                evidence = Evidence::Unsat;
                break;
            }
        }
        (verdict, evidence, true)
    }

    fn run_canfail(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        path: BoolId,
        defcond: BoolId,
    ) -> (Verdict, Evidence, bool) {
        let taken = self.z3.boolean(terms, path);
        let failing = self.z3.boolean(terms, defcond).not();
        let (result, evidence) = {
            let solver = self.solver_for(terms, key.epoch, cone);
            solver.assert(&taken);
            solver.assert(&failing);
            let result = solver.check();
            let evidence = outcome_evidence(solver, result);
            solver.pop(1);
            (result, evidence)
        };
        self.charge(result);
        (Verdict::CanFail(result != SatResult::Unsat), evidence, true)
    }

    /// MAX and MIN through `z3::Optimize`.
    ///
    /// The objective is bounded whenever the interval layer gives a finite
    /// bound on that side, and that bound comes from the `var_range`
    /// constraints, which are in every Φ_e. So an optimum the model reports
    /// is a true optimum. Without a finite interval bound the question is
    /// not asked: an unbounded objective makes `Optimize` report a model
    /// whose value means nothing.
    fn run_bound(
        &mut self,
        terms: &TermArena,
        key: &QuestionKey,
        cone: &Cone,
        term: TermId,
        upper: bool,
    ) -> (Verdict, Evidence, bool) {
        let range = layers::interval(terms, self.phi, term);
        let limit = if upper { range.hi } else { range.lo };
        let Some(limit) = limit else {
            return (
                Verdict::NoBound,
                Evidence::Interval { lo: range.lo, hi: range.hi },
                false,
            );
        };
        let condition = self.condition_assertion(terms, key.condition);
        let expression = self.z3.int(terms, term);
        let assertions = self.assertions(terms, cone);
        let optimizer = z3::Optimize::new();
        optimizer.set_params(&self.params());
        for assertion in &assertions {
            optimizer.assert(assertion);
        }
        if let Some(assertion) = &condition {
            optimizer.assert(assertion);
        }
        if upper {
            optimizer.maximize(&expression);
        } else {
            optimizer.minimize(&expression);
        }
        let result = optimizer.check(&[]);
        let optimum = optimizer
            .get_model()
            .and_then(|model| model.eval(&expression, true))
            .and_then(|value| value.to_string().parse::<i128>().ok());
        let model = optimizer
            .get_model()
            .map(|model| model.to_string())
            .unwrap_or_default();
        self.charge(result);
        self.stats.solver.solvers_built += 1;
        match result {
            SatResult::Sat => match optimum {
                // The interval bound is sound on every admitted run, so it
                // also caps what the optimiser may report.
                Some(value) => {
                    let bounded = if upper { value.min(limit) } else { value.max(limit) };
                    (Verdict::Bound(bounded), Evidence::Model(model), true)
                }
                None => (
                    Verdict::Bound(limit),
                    Evidence::Interval { lo: range.lo, hi: range.hi },
                    false,
                ),
            },
            // Φ_e ∧ g has no model: the value is never built, so any
            // capacity serves. The interval bound is the one to record.
            SatResult::Unsat => (Verdict::Bound(limit), Evidence::Unsat, false),
            SatResult::Unknown => (
                Verdict::Bound(limit),
                Evidence::Interval { lo: range.lo, hi: range.hi },
                false,
            ),
        }
    }

    fn charge(&mut self, result: SatResult) {
        self.stats.solver.queries += 1;
        match result {
            SatResult::Sat => self.stats.solver.sat += 1,
            SatResult::Unsat => self.stats.solver.unsat += 1,
            SatResult::Unknown => self.stats.solver.unknown += 1,
        }
    }

    fn params(&self) -> Params {
        let mut params = Params::new();
        params.set_u32("timeout", self.limits.query_millis);
        params
    }

    /// The constraints of the cone as Z3 expressions.
    fn assertions(&mut self, terms: &TermArena, cone: &Cone) -> Vec<z3::ast::Bool> {
        let mut out = Vec::with_capacity(cone.constraints.len() + cone.products.len());
        for &at in &cone.constraints {
            let formula = self.phi.constraints()[at].formula;
            out.push(self.z3.boolean(terms, formula));
        }
        for &at in &cone.products {
            let fact = &self.phi.products()[at];
            let mut product = arena::integer(1);
            for &factor in &fact.factors {
                product = product * self.z3.int(terms, factor);
            }
            let variable = match self.z3.variable(terms, fact.var) {
                arena::Expr::Int(value) => value,
                // A product fact's variable is `σ`, an integer (§3.4).
                arena::Expr::Bool(_) => continue,
            };
            out.push(variable.eq(product));
        }
        out
    }

    /// The solver for a question at `epoch` over the sliced `cone`, with a
    /// `push` done and the cone's checks enabled; the caller asserts its
    /// condition and goal, checks, and pops.
    ///
    /// The solver itself is cached per connected component of Φ (the cone at
    /// the last epoch), not per (epoch, cone): an unrolled loop gives every
    /// value its own epoch, and one solver per epoch asserted the k checks
    /// before it — quadratic over the program. Here the component is
    /// asserted once, every check behind a guard literal, and a question
    /// enables the guards of the checks in its own epoch's cone. Base
    /// constraints and product facts hold in every Φ_e, so asserting the
    /// whole component's is sound at any epoch.
    fn solver_for(&mut self, terms: &TermArena, epoch: u32, cone: &Cone) -> &Solver {
        let seeds: Vec<u32> = cone
            .constraints
            .iter()
            .map(|&at| self.phi.constraints()[at].formula.index() as u32)
            .collect();
        let mut component = if seeds.is_empty() {
            cone.clone()
        } else {
            self.layers.cone(terms, &seeds, self.layers.check_count())
        };
        component.constraints.retain(|at| !self.redundant.contains(at));
        let key = SolverKey {
            constraints: component.constraints.clone(),
            products: component.products.clone(),
        };
        if self.solvers.contains_key(&key) {
            self.stats.solver.solver_reuse += 1;
        } else {
            let solver = Solver::new();
            solver.set_params(&self.params());
            let mut guards = HashMap::new();
            for &at in &component.constraints {
                let formula = self.phi.constraints()[at].formula;
                let assertion = self.z3.boolean(terms, formula);
                match self.layers.position(at) {
                    Some(position) => {
                        let guard = z3::ast::Bool::new_const(format!("check_{position}"));
                        solver.assert(&guard.implies(&assertion));
                        guards.insert(at, guard);
                    }
                    None => solver.assert(&assertion),
                }
            }
            let products = Cone { constraints: Vec::new(), products: component.products.clone(), variables: 0 };
            for assertion in self.assertions(terms, &products) {
                solver.assert(&assertion);
            }
            self.evict();
            self.solvers.insert(key.clone(), CachedSolver { solver, guards });
            self.stats.solver.solvers_built += 1;
        }
        self.recent.retain(|other| other != &key);
        self.recent.push(key.clone());
        let cached = self.solvers.get(&key).expect("just inserted");
        cached.solver.push();
        for &at in &cone.constraints {
            if self.layers.in_epoch(at, epoch) {
                if let Some(guard) = cached.guards.get(&at) {
                    cached.solver.assert(guard);
                }
            }
        }
        &cached.solver
    }

    // ── Conditions ───────────────────────────────────────────────────

    /// The condition as one Z3 assertion, `None` when it is `true` and there
    /// is nothing to assert.
    fn condition_assertion(
        &mut self,
        terms: &TermArena,
        condition: Condition,
    ) -> Option<z3::ast::Bool> {
        if terms.is_true(condition) {
            return None;
        }
        Some(self.z3.boolean(terms, condition))
    }

    /// Drop the least recently used solver once the cache is full.
    fn evict(&mut self) {
        while self.recent.len() >= self.limits.solver_cache {
            let oldest = self.recent.remove(0);
            self.solvers.remove(&oldest);
        }
    }
}

/// The bound interval arithmetic over the `var_range` constraints alone
/// gives for a MAX or a MIN: the trivial bound of `../../README.md` §11,
/// which the report compares the derived one against. `None` for every other
/// kind of question.
fn interval_bound(terms: &TermArena, phi: &Phi, key: &QuestionKey) -> Option<i128> {
    match key.question {
        Question::Max(term) => layers::interval(terms, phi, term).hi,
        Question::Min(term) => layers::interval(terms, phi, term).lo,
        _ => None,
    }
}

/// Whether a claim is one of the condition's conjuncts.
///
/// The condition is a Boolean term, so `ψ_v ∧ ¬b` holds `¬b` as a conjunct
/// like any other and the negated arm needs no separate treatment
/// (`questions::Condition`).
fn is_conjunct(terms: &TermArena, condition: Condition, claim: BoolId) -> bool {
    layers::conjuncts(terms, condition).contains(&(claim.index() as u32))
}

/// The evidence of one solver result: `unsat`, the model Z3 printed, or why
/// it did not answer (`../../README.md` §5.6).
fn outcome_evidence(solver: &Solver, result: SatResult) -> Evidence {
    match result {
        SatResult::Unsat => Evidence::Unsat,
        SatResult::Sat => match solver.get_model() {
            Some(model) => Evidence::Model(model.to_string()),
            None => Evidence::Model(String::new()),
        },
        SatResult::Unknown => Evidence::Unknown(
            solver
                .get_reason_unknown()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    }
}
