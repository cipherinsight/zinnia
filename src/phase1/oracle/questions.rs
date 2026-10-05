//! The five kinds of question of `revision/README.md` §5.5.
//!
//! A rule turns its obligation list into concrete questions by substituting
//! the descriptor terms of the value it matched. There are five kinds and no
//! more; a new rule brings new instances, not new kinds. Every question is
//! asked under the execution condition `g` of the node that asks it, over
//! the epoch's slice Φ_e of the relation system.
//!
//! `g` is one Boolean term. Phase 1 builds terms over Φ's variables in Φ's
//! term language (`../README.md` §1, "Read-only": it adds no variable and no
//! constraint), so the condition `SELECT` reads its else-arm under, `ψ_v ∧
//! ¬b`, is a term like any other — built through the arena's constructors by
//! [`crate::phase1::build::conjunction`], which orders the conjuncts so that
//! one set of conjuncts is always one term id and the memo keeps its hits.

use crate::phase0::term::{BoolId, TermId};

/// `g`: the Boolean term a question is asked under; `true` when the question
/// is asked unconditionally.
pub type Condition = BoolId;

/// One question, with the term it is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Question {
    /// The maximum of `e` under Φ_e ∧ g: a capacity, a query count, a fold
    /// width.
    Max(TermId),
    /// The minimum of `e`: whether the value can be empty.
    Min(TermId),
    /// Is `P` entailed — is Φ_e ∧ g ∧ ¬P unsatisfiable? A structural
    /// premise, an alignment, an in-range claim: every comparison the
    /// library proves is a built `eq` or `le` term (`rules/README.md` §6.3).
    Prove(BoolId),
    /// Does `e` take only one value under Φ_e ∧ g?
    Const(TermId),
    /// Does `e` take only one Boolean value?
    ConstBool(BoolId),
    /// Has Φ_{position−1} ∧ φ_k ∧ ¬defcond_k a model? Asked once per check.
    CanFail { path: BoolId, defcond: BoolId },
}

impl Question {
    pub fn kind(self) -> Kind {
        match self {
            Question::Max(_) => Kind::Max,
            Question::Min(_) => Kind::Min,
            Question::Prove(_) => Kind::Prove,
            Question::Const(_) | Question::ConstBool(_) => Kind::Const,
            Question::CanFail { .. } => Kind::CanFail,
        }
    }

    /// The arena indices the question is about, for the memo key and for the
    /// dependency cone.
    pub fn nodes(self) -> Vec<u32> {
        match self {
            Question::Max(t) | Question::Min(t) | Question::Const(t) => vec![t.index() as u32],
            Question::Prove(b) | Question::ConstBool(b) => vec![b.index() as u32],
            Question::CanFail { path, defcond } => {
                vec![path.index() as u32, defcond.index() as u32]
            }
        }
    }
}

/// The five kinds, for the statistics and the JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Max,
    Min,
    Prove,
    Const,
    CanFail,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Max, Kind::Min, Kind::Prove, Kind::Const, Kind::CanFail];

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Max => "MAX",
            Kind::Min => "MIN",
            Kind::Prove => "PROVE",
            Kind::Const => "CONST",
            Kind::CanFail => "CANFAIL",
        }
    }
}

/// The memo key of `../README.md` §4: (kind, term, epoch, g). Terms are
/// hash-consed, so the same question asked from two nodes is one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuestionKey {
    pub question: Question,
    pub epoch: u32,
    pub condition: Condition,
}
