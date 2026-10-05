//! The oracle: the layers agree, the cone is sliced, the memo holds.

use crate::phase0::desc::ValueKey;
use crate::phase0::phi::{ConstraintKind, Phi, VarOrigin};
use crate::phase0::term::{BoolId, Sort, TermArena, TermId};

use crate::phase1::oracle::answers::{Layer, Verdict};
use crate::phase1::oracle::questions::Question;
use crate::phase1::oracle::{Limits, Oracle};

/// A hand-built Φ: variables with ranges, plus whatever facts a test adds.
struct Fixture {
    terms: TermArena,
    phi: Phi,
}

fn key() -> ValueKey {
    ValueKey {
        stmt_path: Vec::new(),
        iters: Vec::new(),
        calls: Vec::new(),
        expr_index: 0,
        step: 0,
    }
}

impl Fixture {
    fn new() -> Fixture {
        Fixture { terms: TermArena::new(), phi: Phi::new() }
    }

    fn var(&mut self, name: &str, lo: i128, hi: i128) -> TermId {
        let lo = self.terms.constant(lo);
        let hi = self.terms.constant(hi);
        let id = self.phi.fresh_var(
            &mut self.terms,
            name,
            Sort::Idx,
            VarOrigin::DataScalar { value: key() },
            Some(lo),
            Some(hi),
        );
        self.terms.var(id)
    }

    /// An oracle with the interval layer switched off, so that every
    /// question reaches Z3 and the two layers can be compared, together
    /// with the arena every call lends it and the condition `true`.
    fn solver_only(&self) -> (Oracle<'_>, &TermArena, BoolId) {
        (
            Oracle::new(&self.terms, &self.phi, Limits::default()),
            &self.terms,
            self.terms.tt(),
        )
    }
}

/// The interval layer and Z3 agree on the five kinds.
///
/// The interval layer answers what it can; the same question asked over a Φ
/// the intervals cannot decide reaches Z3 and gives the same verdict.
#[test]
fn the_layers_agree() {
    let mut fixture = Fixture::new();
    let n = fixture.var("n", 0, 1024);
    let c = fixture.var("c", 0, 8);
    let sum = fixture.terms.add(n, c);
    let claim = fixture.terms.le(c, n);
    let bound = fixture.terms.constant(1032);
    let within = fixture.terms.le(sum, bound);
    let tt = fixture.terms.tt();

    let (mut oracle, terms, always) = fixture.solver_only();
    // MAX and MIN: `n + c` mentions each variable once, so the interval is
    // the exact answer (`../../README.md` §5.6).
    assert_eq!(
        oracle.verdict(&terms, Question::Max(sum), 0, always),
        Verdict::Bound(1032)
    );
    assert_eq!(oracle.verdict(&terms, Question::Min(sum), 0, always), Verdict::Bound(0));
    // PROVE: `n + c <= 1032` follows from the ranges; `c <= n` does not.
    assert_eq!(oracle.verdict(&terms, Question::Prove(within), 0, always), Verdict::Proved);
    assert_eq!(
        oracle.verdict(&terms, Question::Prove(claim), 0, always),
        Verdict::NotProved
    );
    // CONST: the sum is not constant.
    assert_eq!(oracle.verdict(&terms, Question::Const(sum), 0, always), Verdict::NotConstant);
    // CANFAIL: `c <= n` can fail under the ranges alone.
    assert_eq!(
        oracle.verdict(&terms, Question::CanFail { path: tt, defcond: claim }, 0, always),
        Verdict::CanFail(true)
    );
}

/// A question whose answer needs a `precondition` reaches Z3, and Z3 agrees
/// with what the intervals would say when they can say it.
#[test]
fn a_precondition_is_read_by_the_solver_only() {
    let mut fixture = Fixture::new();
    let n = fixture.var("n", 0, 1024);
    let eight = fixture.terms.constant(8);
    let fact = fixture.terms.le(n, eight);
    fixture
        .phi
        .add_fact(&mut fixture.terms, fact, ConstraintKind::Precondition, "test");

    let (mut oracle, terms, always) = fixture.solver_only();
    // The interval bound is 1024; the precondition makes the true maximum 8,
    // and only the solver sees it.
    let answer = oracle.ask(&terms, Question::Max(n), 0, always);
    let recorded = oracle.answers.get(answer).expect("just asked");
    assert_eq!(recorded.verdict, Verdict::Bound(8));
    assert_eq!(recorded.layer, Layer::Solver);
    assert_eq!(oracle.stats.solver.queries, 1);
}

/// A Z3 query asserts the constraints of the cone and nothing else.
#[test]
fn a_query_asserts_only_its_cone() {
    let mut fixture = Fixture::new();
    // Two independent halves of Φ: the question is about the first, so the
    // second must not be asserted.
    let n = fixture.var("n", 0, 1024);
    let m = fixture.var("m", 0, 1024);
    let eight = fixture.terms.constant(8);
    let about_n = fixture.terms.le(n, eight);
    let about_m = fixture.terms.le(m, eight);
    fixture
        .phi
        .add_fact(&mut fixture.terms, about_n, ConstraintKind::Precondition, "n");
    fixture
        .phi
        .add_fact(&mut fixture.terms, about_m, ConstraintKind::Precondition, "m");

    let (mut oracle, terms, always) = fixture.solver_only();
    let answer = oracle.ask(&terms, Question::Max(n), 0, always);
    let recorded = oracle.answers.get(answer).expect("just asked");
    assert_eq!(recorded.verdict, Verdict::Bound(8));
    // Φ holds four constraints — two ranges and two preconditions — and the
    // cone of `n` holds two.
    assert_eq!(oracle.stats.solver.phi_constraints, 4);
    assert_eq!(recorded.cone_constraints, 2);
    assert_eq!(recorded.cone_variables, 1);
}

/// The same question from two nodes is one entry in A and one solver call.
#[test]
fn the_memo_answers_the_second_ask() {
    let mut fixture = Fixture::new();
    let n = fixture.var("n", 0, 1024);
    let eight = fixture.terms.constant(8);
    let fact = fixture.terms.le(n, eight);
    fixture
        .phi
        .add_fact(&mut fixture.terms, fact, ConstraintKind::Precondition, "test");

    let (mut oracle, terms, always) = fixture.solver_only();
    let first = oracle.ask(&terms, Question::Max(n), 0, always);
    let second = oracle.ask(&terms, Question::Max(n), 0, always);
    assert_eq!(first, second);
    assert_eq!(oracle.answers.len(), 1);
    assert_eq!(oracle.stats.solver.queries, 1);
}

/// A solver is built once per (epoch, cone) and reused for the next question
/// over the same cone.
#[test]
fn a_cone_keeps_its_solver() {
    let mut fixture = Fixture::new();
    let n = fixture.var("n", 0, 1024);
    let eight = fixture.terms.constant(8);
    let fact = fixture.terms.le(n, eight);
    fixture
        .phi
        .add_fact(&mut fixture.terms, fact, ConstraintKind::Precondition, "test");
    let four = fixture.terms.constant(4);
    let claim = fixture.terms.le(n, four);
    let other = fixture.terms.le(n, eight);

    let (mut oracle, terms, always) = fixture.solver_only();
    assert_eq!(oracle.verdict(&terms, Question::Prove(claim), 0, always), Verdict::NotProved);
    assert_eq!(oracle.verdict(&terms, Question::Prove(other), 0, always), Verdict::Proved);
    assert_eq!(oracle.stats.solver.solvers_built, 1);
    assert!(oracle.stats.solver.solver_reuse >= 1);
}

/// The interval layer and Z3 give the same verdict for all five kinds.
///
/// Each kind is asked twice: once over a Φ whose ranges decide it, where the
/// interval layer answers, and once over a Φ that states the same fact as a
/// `precondition`, where only the solver sees it. The verdicts agree and the
/// layers differ, which is what "cheapest layer first"
/// (`../../README.md` §5.6) has to mean.
#[test]
fn the_two_layers_give_the_same_verdict() {
    use crate::phase0::term::VarId;

    /// The one variable of the fixture.
    fn n(fixture: &mut Fixture) -> TermId {
        fixture.terms.var(VarId(0))
    }

    fn at_most_eight(fixture: &mut Fixture) -> BoolId {
        let variable = n(fixture);
        let eight = fixture.terms.constant(8);
        fixture.terms.le(variable, eight)
    }

    fn is_five(fixture: &mut Fixture) -> BoolId {
        let variable = n(fixture);
        let five = fixture.terms.constant(5);
        fixture.terms.eq(variable, five)
    }

    fn at_least_three(fixture: &mut Fixture) -> BoolId {
        let variable = n(fixture);
        let three = fixture.terms.constant(3);
        fixture.terms.le(three, variable)
    }

    fn max(fixture: &mut Fixture) -> Question {
        Question::Max(n(fixture))
    }

    fn min(fixture: &mut Fixture) -> Question {
        Question::Min(n(fixture))
    }

    fn prove(fixture: &mut Fixture) -> Question {
        Question::Prove(at_most_eight(fixture))
    }

    fn constant(fixture: &mut Fixture) -> Question {
        Question::Const(n(fixture))
    }

    fn canfail(fixture: &mut Fixture) -> Question {
        let path = fixture.terms.tt();
        Question::CanFail { path, defcond: at_most_eight(fixture) }
    }

    /// One question over one Φ, with the layer that answered it.
    fn ask(
        range: (i128, i128),
        fact: Option<fn(&mut Fixture) -> BoolId>,
        question: fn(&mut Fixture) -> Question,
    ) -> (Verdict, Layer) {
        let mut fixture = Fixture::new();
        fixture.var("n", range.0, range.1);
        if let Some(fact) = fact {
            let formula = fact(&mut fixture);
            fixture.phi.add_fact(
                &mut fixture.terms,
                formula,
                ConstraintKind::Precondition,
                "test",
            );
        }
        let asked = question(&mut fixture);
        let (mut oracle, terms, always) = fixture.solver_only();
        let at = oracle.ask(&terms, asked, 0, always);
        let answer = oracle.answers.get(at).expect("just asked");
        (answer.verdict.clone(), answer.layer)
    }

    // MAX: the range says 8, or the precondition does.
    assert_eq!(ask((0, 8), None, max), (Verdict::Bound(8), Layer::Interval));
    assert_eq!(
        ask((0, 1024), Some(at_most_eight), max),
        (Verdict::Bound(8), Layer::Solver)
    );
    // MIN: the same, from below.
    assert_eq!(ask((3, 10), None, min), (Verdict::Bound(3), Layer::Interval));
    assert_eq!(
        ask((0, 10), Some(at_least_three), min),
        (Verdict::Bound(3), Layer::Solver)
    );
    // PROVE `n <= 8`.
    assert_eq!(ask((0, 8), None, prove), (Verdict::Proved, Layer::Interval));
    assert_eq!(
        ask((0, 1024), Some(at_most_eight), prove),
        (Verdict::Proved, Layer::Solver)
    );
    // CONST `n`.
    assert_eq!(
        ask((5, 5), None, constant),
        (Verdict::Constant(5), Layer::Interval)
    );
    assert_eq!(
        ask((0, 1024), Some(is_five), constant),
        (Verdict::Constant(5), Layer::Solver)
    );
    // CANFAIL of the check `true ⇒ n <= 8`.
    assert_eq!(
        ask((0, 8), None, canfail),
        (Verdict::CanFail(false), Layer::Interval)
    );
    assert_eq!(
        ask((0, 1024), Some(at_most_eight), canfail),
        (Verdict::CanFail(false), Layer::Solver)
    );
}
