//! A, the answer record of `revision/phase1/README.md` §1 and §4.
//!
//! Every question a node asked is in A with its evidence, and A holds no
//! other question (§2, "Invariants"). The record doubles as the memo: a
//! question is looked up by (kind, term, epoch, g) before it is asked, so a
//! term shared by many nodes is asked once.

use std::collections::HashMap;

use super::questions::{Kind, Question, QuestionKey};

/// Index of an answer in [`Answers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnswerId(pub u32);

/// Which layer answered (`../README.md` §3.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// The term is literally a constant, `true`, or a conjunct of the
    /// context.
    Syntactic,
    /// Interval arithmetic over the `var_range` constraints.
    Interval,
    Solver,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::Syntactic => "syntactic",
            Layer::Interval => "interval",
            Layer::Solver => "z3",
        }
    }
}

/// What the oracle concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// PROVE: Φ_e ∧ g ∧ ¬P has no model.
    Proved,
    /// PROVE: not proved, whether refuted or unknown.
    NotProved,
    /// CONST: the one value.
    Constant(i128),
    ConstantBool(bool),
    NotConstant,
    /// MAX / MIN.
    Bound(i128),
    /// MAX / MIN: no bound the oracle could establish.
    NoBound,
    /// CANFAIL: the check can fail (or the oracle does not know it cannot).
    CanFail(bool),
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Proved => "proved",
            Verdict::NotProved => "not_proved",
            Verdict::Constant(_) => "constant",
            Verdict::ConstantBool(_) => "constant_bool",
            Verdict::NotConstant => "not_constant",
            Verdict::Bound(_) => "bound",
            Verdict::NoBound => "no_bound",
            Verdict::CanFail(true) => "can_fail",
            Verdict::CanFail(false) => "cannot_fail",
        }
    }

    pub fn value(&self) -> Option<i128> {
        match self {
            Verdict::Constant(n) | Verdict::Bound(n) => Some(*n),
            Verdict::ConstantBool(b) => Some(*b as i128),
            _ => None,
        }
    }
}

/// The evidence phase 3 replays (`../../README.md` §5.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evidence {
    /// The term was literally a constant or `true`, or `g` contained it as a
    /// conjunct.
    Syntactic(String),
    /// The interval derivation: the interval of the term over the ranges.
    Interval { lo: Option<i128>, hi: Option<i128> },
    /// `unsat` for a PROVE, or the model for a MAX or a satisfiable CANFAIL.
    Unsat,
    /// The solver's model, as Z3 prints it.
    Model(String),
    /// The solver did not answer within the time limit.
    Unknown(String),
}

/// One answer, with everything §5.6 says it records.
#[derive(Debug, Clone)]
pub struct Answer {
    pub key: QuestionKey,
    pub layer: Layer,
    pub verdict: Verdict,
    pub evidence: Evidence,
    /// Whether the bound is known to be attained (`../../README.md` §5.6,
    /// tightness). The interval layer marks a bound tight only when it is
    /// exact by construction.
    pub tight: bool,
    /// Milliseconds the solver spent, 0 for the cheap layers.
    pub millis: u128,
    /// Constraints and variables the solver was given, for the scalability
    /// report (§3.2.6).
    pub cone_constraints: usize,
    pub cone_variables: usize,
    /// For a MAX or a MIN, the bound interval arithmetic over the ranges
    /// alone gives: the *trivial* bound of `../../README.md` §11, against
    /// which the answer is the derived one. `None` for every other kind and
    /// where the intervals give no bound on that side.
    pub interval_bound: Option<i128>,
}

impl Answer {
    pub fn kind(&self) -> Kind {
        self.key.question.kind()
    }
}

/// A: the answers, in the order they were asked, with the memo over them.
#[derive(Debug, Default)]
pub struct Answers {
    entries: Vec<Answer>,
    memo: HashMap<QuestionKey, AnswerId>,
    /// The answers no implementation node names: the CANFAIL sweep of §4
    /// step 1, which decides the roots before any node exists, and the
    /// premises of a rule instance that then found no provider for a port.
    unattached: Vec<AnswerId>,
}

impl Answers {
    pub fn lookup(&self, key: &QuestionKey) -> Option<AnswerId> {
        self.memo.get(key).copied()
    }

    pub fn record(&mut self, answer: Answer) -> AnswerId {
        let key = answer.key;
        let at = AnswerId(self.entries.len() as u32);
        self.entries.push(answer);
        self.memo.insert(key, at);
        at
    }

    /// Mark an answer as one the construction asked rather than a node: the
    /// CANFAIL sweep, or a premise of a rule instance that produced no node.
    pub fn mark_root(&mut self, at: AnswerId) {
        self.unattached.push(at);
    }

    /// The answers no node names, which the invariant checker admits.
    pub fn roots(&self) -> Vec<AnswerId> {
        self.unattached.clone()
    }

    pub fn get(&self, at: AnswerId) -> Option<&Answer> {
        self.entries.get(at.0 as usize)
    }

    pub fn ids(&self) -> impl Iterator<Item = AnswerId> {
        (0..self.entries.len() as u32).map(AnswerId)
    }

    pub fn entries(&self) -> &[Answer] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The question the oracle is about to ask, when it has been asked
    /// before.
    pub fn question_of(&self, at: AnswerId) -> Option<Question> {
        self.get(at).map(|answer| answer.key.question)
    }
}
