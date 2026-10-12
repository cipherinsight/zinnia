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

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// The shared operands (the operator's inputs). An element group's are
    /// the cells it reads, when they are statements of the graph (an
    /// `ArrayInit`'s operands).
    pub shared: Vec<StmtId>,
    /// The live lanes, in statement order.
    pub lanes: Vec<Lane>,
    /// For an element group that reads a written array version: the
    /// version and the addresses of the cells its reads can name (its
    /// `shared` is empty). The version's memory group produces those cells
    /// (`Requirement::SharedAt { at: version, name: memory::versions::CELLS }`),
    /// which every candidate of this group producing its lanes needs
    /// (`emit::lower` adds the precondition).
    pub version: Option<VersionLine>,
    /// For a memory group (`OpKind::Versions`): its version tree.
    pub tree: Option<std::rc::Rc<crate::lower::memory::versions::Tree>>,
}

/// The cells of a written array version an element group reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VersionLine {
    pub version: StmtId,
    pub addresses: Vec<usize>,
}

/// Where an element group's cells come from: statements of the graph, or a
/// written version's cells (`EmitCx::cells` gives their values).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellSource {
    Stmts(Vec<StmtId>),
    Version(VersionLine),
}

impl CellSource {
    pub fn len(&self) -> usize {
        match self {
            CellSource::Stmts(cells) => cells.len(),
            CellSource::Version(line) => line.addresses.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Group {
    /// The number of cells an element group reads.
    pub fn line_len(&self) -> usize {
        self.version.as_ref().map_or(self.shared.len(), |line| line.addresses.len())
    }

    /// Where an element group's cells come from.
    pub fn cells(&self) -> CellSource {
        match &self.version {
            Some(line) => CellSource::Version(line.clone()),
            None => CellSource::Stmts(self.shared.clone()),
        }
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
    /// A value computed from the statement `at` alone (the bits of an index,
    /// say) that nodes of several groups may share, produced once, named by
    /// its family.
    SharedAt { at: StmtId, name: &'static str },
    /// A capability of the whole program, produced at most once whatever
    /// groups use it: the verifier challenge (`challenge::CHALLENGE`). Its
    /// producer is the framework's (`challenge::producer_cost`), with a
    /// one-time cost per target, or none where the target cannot provide
    /// it; the selection pays it once, for every part of the graph together.
    Program { name: &'static str },
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
    /// The two lists of values are equal as multisets (`is_permutation(p)`
    /// is `Permutation(p, [0, 1, ..., n - 1])`).
    Permutation(ListId, ListId),
    /// `x <= y`, or `x < y` when `strict`, for two integers or two floats'
    /// fixed-point codes (the codes are ordered as the floats).
    Ordered { x: StmtId, y: StmtId, strict: bool },
    /// The Boolean statement's value is true (what `Assert(c)` states; the
    /// basis of a fact an `IR::ArrayFact` statement states).
    Holds(StmtId),
    // ── Array claims (`super::array_facts`): over the values of a list ──
    /// Every value of the list lies in `[lo, hi]` (`None`: an open end).
    /// `NonNegativeElements(list)` is `ElementsWithin { list, lo: Some(0),
    /// hi: None }`.
    ElementsWithin { list: ListId, lo: Option<i64>, hi: Option<i64> },
    /// `list[0] <= list[1] <= ...`.
    NonDecreasing(ListId),
    /// `list[0] < list[1] < ...`.
    StrictlyIncreasing(ListId),
    /// No two values of the list are equal.
    Distinct(ListId),
    /// At most `max_nonzero` values of the list are nonzero.
    CountBound { list: ListId, max_nonzero: u32 },
}

/// A list of values a claim is about, interned (claims are small `Copy`
/// values): equal lists have equal ids, so comparing ids compares lists.
///
/// The interner belongs to one lowering: [`ListId::reset`] empties it when
/// `emit::lower` starts, and it is per thread (a lowering runs on one).
/// Ids are therefore numbered in the order one compilation creates its
/// lists, whatever the process compiled before, so the order of claims
/// (`ListId` is `Ord`) is reproducible, and the table does not grow across
/// compilations. An id is meaningful only within the lowering that made it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ListId(u32);

/// One value of a [`ListId`]'s list: a statement's value or an integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ListEntry {
    Stmt(StmtId),
    Int(i64),
}

/// The lists by id, the ids by list, and each list's statement positions
/// (`ListId::positions`, built on demand).
type Interned = (Vec<Arc<[ListEntry]>>, HashMap<Arc<[ListEntry]>, u32>, HashMap<u32, Arc<HashMap<StmtId, Vec<usize>>>>);

thread_local! {
    static LISTS: RefCell<Interned> = RefCell::new(Default::default());
}

impl ListId {
    /// Forget every list (the start of a lowering, `emit::lower`).
    pub(super) fn reset() {
        LISTS.with(|lists| *lists.borrow_mut() = Default::default());
    }

    pub fn intern(entries: &[ListEntry]) -> ListId {
        LISTS.with(|lists| Self::intern_in(&mut lists.borrow_mut(), entries))
    }

    fn intern_in(lists: &mut Interned, entries: &[ListEntry]) -> ListId {
        if let Some(&id) = lists.1.get(entries) {
            return ListId(id);
        }
        let list: Arc<[ListEntry]> = entries.into();
        let id = lists.0.len() as u32;
        lists.0.push(list.clone());
        lists.1.insert(list, id);
        ListId(id)
    }

    /// The statements of a list of statements.
    pub fn of_stmts(stmts: &[StmtId]) -> ListId {
        ListId::intern(&stmts.iter().map(|&s| ListEntry::Stmt(s)).collect::<Vec<_>>())
    }

    /// `[0, 1, ..., n - 1]`.
    pub fn range(n: usize) -> ListId {
        ListId::intern(&(0..n as i64).map(ListEntry::Int).collect::<Vec<_>>())
    }

    pub fn entries(self) -> Arc<[ListEntry]> {
        LISTS.with(|lists| lists.borrow().0[self.0 as usize].clone())
    }

    pub fn len(self) -> usize {
        self.entries().len()
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// The statements among the list's values, in list order.
    pub fn stmts(self) -> Vec<StmtId> {
        self.entries()
            .iter()
            .filter_map(|entry| match *entry {
                ListEntry::Stmt(s) => Some(s),
                ListEntry::Int(_) => None,
            })
            .collect()
    }

    /// The positions of each statement in the list (computed once per list
    /// and lowering).
    pub fn positions(self) -> Arc<HashMap<StmtId, Vec<usize>>> {
        LISTS.with(|lists| {
            let mut lists = lists.borrow_mut();
            if let Some(index) = lists.2.get(&self.0) {
                return index.clone();
            }
            let mut index: HashMap<StmtId, Vec<usize>> = HashMap::new();
            for (k, entry) in lists.0[self.0 as usize].iter().enumerate() {
                if let ListEntry::Stmt(s) = entry {
                    index.entry(*s).or_default().push(k);
                }
            }
            let index = Arc::new(index);
            lists.2.insert(self.0, index.clone());
            index
        })
    }

    /// Whether the statement `s` is one of the list's values.
    pub fn contains(self, s: StmtId) -> bool {
        self.positions().contains_key(&s)
    }
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
    /// statements, a narrower constant range; for array claims, see
    /// `super::array_facts::implies`).
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
            (Within { x, lo, .. }, NonNegative(y)) => x == y && lo.is_some_and(|l| l >= 0),
            (NonNegative(x), Within { x: y, lo, hi: None }) if super::array_facts::enabled() => {
                x == y && lo.is_none_or(|l| l <= 0)
            }
            (Within { x, lo, hi }, NonZero(y)) => x == y && (lo.is_some_and(|l| l >= 1) || hi.is_some_and(|h| h <= -1)),
            (Equal(a, b), Equal(c, d)) => (a, b) == (c, d) || (a, b) == (d, c),
            (Ordered { x, y, strict }, Ordered { x: x2, y: y2, strict: strict2 }) => {
                (x, y) == (x2, y2) && (strict || !strict2)
            }
            (a, b) if a.is_array_claim() || matches!(a, Permutation(..)) => super::array_facts::implies(&a, &b),
            (a, b) => a == b,
        }
    }

    /// Whether the claim is about a list's values as a whole (the claims of
    /// `super::array_facts`; `Permutation` relates two lists).
    pub fn is_array_claim(&self) -> bool {
        matches!(
            self,
            Claim::ElementsWithin { .. }
                | Claim::NonDecreasing(_)
                | Claim::StrictlyIncreasing(_)
                | Claim::Distinct(_)
                | Claim::CountBound { .. }
        )
    }

    /// The statements the claim is about.
    pub fn subjects(&self) -> Vec<StmtId> {
        match *self {
            Claim::InRange { x, hi: Bound::Stmt(h), .. } => vec![x, h],
            Claim::InRange { x, .. } | Claim::NonNegative(x) | Claim::NonZero(x) | Claim::Within { x, .. } => vec![x],
            Claim::Equal(a, b) | Claim::Ordered { x: a, y: b, .. } => vec![a, b],
            Claim::Holds(c) => vec![c],
            Claim::Permutation(a, b) => [a, b].iter().flat_map(|list| list.stmts()).collect(),
            Claim::ElementsWithin { list, .. }
            | Claim::NonDecreasing(list)
            | Claim::StrictlyIncreasing(list)
            | Claim::Distinct(list)
            | Claim::CountBound { list, .. } => list.stmts(),
        }
    }
}
