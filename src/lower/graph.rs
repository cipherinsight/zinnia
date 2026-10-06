//! The vocabulary of the layer: operator groups, requirement nodes and
//! implementation nodes.
//!
//! * A **requirement** is something the compiled program must have: the
//!   value of a live lane, a value shared by several constructions of one
//!   group (the elements in a memory segment), or a **fact** (an obligation
//!   such as "`i` lies in `[0, n)`" where some predicate holds).
//! * An **implementation node** is one way to produce some requirements: a
//!   [`super::Candidate`] a construction offers, a program check (the
//!   assertion the program already contains), or a free fact.
//!
//! Statement ids in this module refer to the graph being lowered (the input
//! of the pass).

use crate::types::StmtId;

use super::op::OpKind;

/// Index of a group in the pass's list of groups.
pub type GroupId = usize;

/// One live output lane of an operator: its placeholder statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    /// The `IR::ArrayOp` statement.
    pub stmt: StmtId,
    /// Which output of the operator (see `ArrayOp::output`).
    pub output: u8,
    /// The position in that output, `None` at a runtime position.
    pub position: Option<u32>,
    /// The lane's own operands (after the shared ones).
    pub own: Vec<StmtId>,
}

/// One operator instance: its kind, its inputs and its live lanes.
#[derive(Debug, Clone)]
pub struct Group {
    pub id: GroupId,
    pub kind: OpKind,
    /// The shared operands (the operator's inputs).
    pub shared: Vec<StmtId>,
    /// The live lanes, in statement order.
    pub lanes: Vec<Lane>,
}

impl Group {
    /// The live lanes of output `output` at constant positions, by position.
    pub fn positions(&self, output: u8) -> impl Iterator<Item = (u32, &Lane)> {
        self.lanes
            .iter()
            .filter(move |lane| lane.output == output)
            .filter_map(|lane| Some((lane.position?, lane)))
    }

    /// The live lanes of output `output` at runtime positions.
    pub fn runtime_lanes(&self, output: u8) -> impl Iterator<Item = &Lane> {
        self.lanes.iter().filter(move |lane| lane.output == output && lane.position.is_none())
    }
}

/// Something the compiled program must have.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Requirement {
    /// The value of a live lane (its placeholder statement).
    Lane(StmtId),
    /// A value several constructions of one group may share, produced once
    /// (e.g. the group's elements written to a memory segment), named by
    /// its family.
    Shared { group: GroupId, name: &'static str },
    /// A fact.
    Fact(Fact),
}

/// A claim that must hold (or is established) where `scope` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fact {
    pub claim: Claim,
    pub scope: Scope,
}

/// Where a fact holds: on every path, or where a Boolean statement is true
/// (an execution predicate, see `builder/guard.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    Everywhere,
    When(StmtId),
}

/// An integer bound: a constant or a statement's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bound {
    Const(i64),
    Stmt(StmtId),
}

/// The claims facts are made of, over statements' values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Claim {
    /// `lo <= x < hi` for the integer `x`.
    InRange { x: StmtId, lo: i64, hi: Bound },
    /// `x >= 0` (integer, or a float's fixed-point encoding).
    NonNegative(StmtId),
    /// `x != 0`.
    NonZero(StmtId),
    /// `x == y`.
    Equal(StmtId, StmtId),
    /// `lo <= x <= hi` for the integer `x`, both ends inclusive; `None` is
    /// an open end (an asserted `x >= c` or `x <= c`, such as one side of an
    /// input's int64 window, bounds one side only).
    Within { x: StmtId, lo: Option<i64>, hi: Option<i64> },
}

impl Fact {
    pub fn everywhere(claim: Claim) -> Fact {
        Fact { claim, scope: Scope::Everywhere }
    }

    /// The statements the fact is about, its scope's predicate included.
    pub fn subjects(&self) -> Vec<StmtId> {
        let mut subjects = self.claim.subjects();
        if let Scope::When(p) = self.scope {
            subjects.push(p);
        }
        subjects
    }

    /// `true` when `self` established implies `other`: the scope of `self`
    /// covers `other`'s, and its claim implies `other`'s.
    pub fn implies(&self, other: &Fact) -> bool {
        let scope_covers = self.scope == Scope::Everywhere || self.scope == other.scope;
        scope_covers && self.claim.implies(&other.claim)
    }
}

impl Claim {
    /// `true` when `self` implies `other` syntactically (the same
    /// statements, a narrower constant range).
    pub fn implies(&self, other: &Claim) -> bool {
        use Claim::*;
        match (*self, *other) {
            (InRange { x, lo, hi }, InRange { x: y, lo: lo2, hi: hi2 }) if x == y => {
                lo >= lo2
                    && match (hi, hi2) {
                        (Bound::Const(a), Bound::Const(b)) => a <= b,
                        (a, b) => a == b,
                    }
            }
            (InRange { x, lo, hi: Bound::Const(h) }, Within { x: y, lo: lo2, hi: hi2 }) if x == y => {
                lo2.is_none_or(|l| lo >= l) && hi2.is_none_or(|u| h - 1 <= u)
            }
            (Within { x, lo, hi }, Within { x: y, lo: lo2, hi: hi2 }) if x == y => {
                lo2.is_none_or(|l| lo.is_some_and(|lo| lo >= l)) && hi2.is_none_or(|u| hi.is_some_and(|hi| hi <= u))
            }
            (InRange { x, lo, .. }, NonNegative(y)) => x == y && lo >= 0,
            (InRange { x, lo, .. }, NonZero(y)) => x == y && lo >= 1,
            (Equal(a, b), Equal(c, d)) => (a, b) == (c, d) || (a, b) == (d, c),
            (a, b) => a == b,
        }
    }

    /// The statements the claim is about.
    pub fn subjects(&self) -> Vec<StmtId> {
        match *self {
            Claim::InRange { x, hi: Bound::Stmt(h), .. } => vec![x, h],
            Claim::InRange { x, .. } | Claim::NonNegative(x) | Claim::NonZero(x) | Claim::Within { x, .. } => vec![x],
            Claim::Equal(a, b) => vec![a, b],
        }
    }
}
