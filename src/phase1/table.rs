//! The provider table of `revision/phase1/README.md` §4.
//!
//! The construction fills a table whose rows are a construction device, not
//! nodes of G. A row is keyed by a value or term, an interface family, an
//! execution condition, a sharing identity and the epoch filter; its content
//! is the concrete requirement nodes of that family that some implementation
//! node can provide. A consumer cannot name a concrete interface itself —
//! the capacity inside it is the provider's, decided upward through a MAX
//! answer — so it asks for a row and instantiates one node per concrete
//! requirement the row holds.

use std::collections::HashMap;

use crate::phase0::term::BoolId;

use crate::phase1::graph::{EpochFilter, ReqId, SharingId, Target};
use crate::phase1::interface::{Batch, Family, Interface, RanksOf};

/// The memo key of §4, together with the epoch the row's nodes ask their
/// questions at. The epoch is a function of the rest — the value's check
/// watermark, capped by the epoch filter — so it is carried, not
/// keyed on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowRequest {
    pub target: Target,
    pub family: Family,
    /// `g`, the condition the row's providers satisfy their contract under.
    pub condition: BoolId,
    pub sharing: SharingId,
    pub filter: EpochFilter,
    /// The epoch the row's nodes ask their questions at (§4, "Checks and
    /// epochs").
    pub epoch: u32,
    /// The batch, for a `Fetch` row (§4 step 4). A `Represent` row has none;
    /// two `Fetch` rows over one value with different batches are two rows,
    /// because the batch is part of the requirement's arguments
    /// (`../README.md` §2).
    pub batch: Option<Batch>,
    /// `A`, for a `Groups` row: "A is part of the row key"
    /// (`rules/README.md` §4.5). Two axis sets over one value are two rows.
    pub axes: Option<Vec<usize>>,
    /// `d`, for an `IndexNorm` row: §4.3 writes the requirement over the pair
    /// `⟨i, d⟩`, so the extent is part of what the row is about.
    pub extent: Option<crate::phase0::term::TermId>,
    /// `I`, for a `Ranks` row: the identity of `Ranks(I)` is
    /// (mask identity, I) (`rules/README.md` §4.5), and the mask identity is
    /// the row's target. The bundle is therefore part of the key: one value
    /// held in two bundles has two rank chains, and `Ranks(I)` and
    /// `Ranks(I, valid)` over one bundle are two rows.
    pub ranks: Option<RanksOf>,
}

impl RowRequest {
    /// The part of the request the memo is keyed on.
    pub fn key(&self) -> MemoKey {
        MemoKey {
            target: self.target.clone(),
            family: self.family,
            condition: self.condition,
            sharing: self.sharing,
            filter: self.filter.clone(),
            batch: self.batch.clone(),
            axes: self.axes.clone(),
            extent: self.extent,
            ranks: self.ranks.clone(),
        }
    }

    /// The same request for another target and family. The epoch filter
    /// travels unchanged, because it reaches every row below an `Enforce`;
    /// acyclicity is the in-progress stack's and carries nothing
    /// ([`crate::phase1::graph::EpochFilter`]).
    pub fn child(&self, target: Target, family: Family, condition: BoolId, epoch: u32) -> RowRequest {
        RowRequest {
            target,
            family,
            condition,
            sharing: self.sharing,
            filter: self.filter.clone(),
            epoch,
            batch: None,
            axes: None,
            extent: None,
            ranks: None,
        }
    }

    /// The child row a `Groups` port asks for: the fibers of one value over
    /// one axis set (`rules/README.md` §4.5).
    /// The child row an `IndexNorm` port asks for: the pair `⟨i, d⟩` of
    /// `rules/README.md` §4.3.
    pub fn child_index_norm(
        &self,
        index: Target,
        extent: crate::phase0::term::TermId,
        condition: BoolId,
        epoch: u32,
    ) -> RowRequest {
        RowRequest {
            extent: Some(extent),
            ..self.child(index, Family::IndexNorm, condition, epoch)
        }
    }

    /// The child row a `Ranks` port asks for: the rank chain of one bundle
    /// of one value (`rules/README.md` §4.5).
    pub fn child_ranks(
        &self,
        value: Target,
        subject: RanksOf,
        condition: BoolId,
        epoch: u32,
    ) -> RowRequest {
        RowRequest {
            ranks: Some(subject),
            ..self.child(value, Family::Ranks, condition, epoch)
        }
    }

    pub fn child_groups(
        &self,
        value: Target,
        axes: Vec<usize>,
        condition: BoolId,
        epoch: u32,
    ) -> RowRequest {
        RowRequest {
            axes: Some(axes),
            ..self.child(value, Family::Groups, condition, epoch)
        }
    }

    /// The child row a `Fetch` port asks for: the `Values` family over one
    /// batch of the value (§4 step 4).
    pub fn child_fetch(
        &self,
        value: Target,
        batch: Batch,
        condition: BoolId,
        epoch: u32,
    ) -> RowRequest {
        RowRequest {
            batch: Some(batch),
            ..self.child(value, Family::Values, condition, epoch)
        }
    }
}

/// The memo key of §4.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoKey {
    pub target: Target,
    pub family: Family,
    pub condition: BoolId,
    pub sharing: SharingId,
    pub filter: EpochFilter,
    pub batch: Option<Batch>,
    pub axes: Option<Vec<usize>>,
    pub extent: Option<crate::phase0::term::TermId>,
    pub ranks: Option<RanksOf>,
}

/// One concrete requirement a row holds.
#[derive(Debug, Clone)]
pub struct RowEntry {
    pub interface: Interface,
    pub requirement: ReqId,
    /// The epoch of the nodes that provide it.
    pub epoch: u32,
}

/// The content of one row.
#[derive(Debug, Clone, Default)]
pub struct Row {
    pub entries: Vec<RowEntry>,
    /// Why the row is empty, when it is: the missing rule or the premise
    /// that failed, for the rejection message.
    pub notes: Vec<String>,
    /// This is not a row but the answer "the row you ask for is already
    /// being filled, so asking it again closes a cycle" (§4, "Acyclicity by
    /// construction"). It holds no entry, so the candidate whose port asked
    /// for it is dropped, and the row that candidate belonged to becomes
    /// provisional.
    pub cut: bool,
    /// Whether this fill dropped at least one candidate to the cut above.
    /// A provisional row is not memoised and its requirement nodes are its
    /// own, so a later consumer outside the cycle refills it unrestricted
    /// and gets every provider on nodes of its own.
    pub provisional: bool,
}

impl Row {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The table, with the in-progress marks that keep the recursion finite.
#[derive(Debug, Default)]
pub struct Table {
    rows: HashMap<MemoKey, Row>,
    filling: Vec<MemoKey>,
}

impl Table {
    pub fn get(&self, key: &MemoKey) -> Option<&Row> {
        self.rows.get(key)
    }

    /// A row of the same key whose condition is weaker than `condition`
    /// (§4, "Conditions of child requirements"): syntactically, when every
    /// conjunct of the row's condition is a conjunct of `condition`.
    ///
    /// The syntactic test is the one §4 states; a `PROVE(g ⇒ ψ)` fallback is
    /// the caller's, which has the oracle.
    pub fn weaker(
        &self,
        key: &MemoKey,
        conjuncts: &std::collections::BTreeSet<u32>,
        weaker_than: impl Fn(BoolId) -> bool,
    ) -> Option<(&MemoKey, &Row)> {
        let _ = conjuncts;
        self.rows.iter().find(|(other, _)| {
            other.target == key.target
                && other.family == key.family
                && other.sharing == key.sharing
                && other.filter == key.filter
                && other.batch == key.batch
                && other.axes == key.axes
                && other.extent == key.extent
                && other.ranks == key.ranks
                && other.condition != key.condition
                && weaker_than(other.condition)
        })
    }

    pub fn insert(&mut self, key: MemoKey, row: Row) {
        self.rows.insert(key, row);
    }

    /// Whether the row is already being filled, which is how a port that
    /// asks for a row above it in the expansion stops instead of recurring
    /// (§4, "Acyclicity by construction").
    pub fn is_filling(&self, key: &MemoKey) -> bool {
        self.filling.contains(key)
    }

    /// Whether a row about this target and family is being filled, whatever
    /// its condition and epoch filter.
    ///
    /// `rules/README.md` §3.2 makes the row of a term that is the term of a
    /// value of Δ that value's row. Inside that value's own row the term has
    /// to be compiled instead, which is `LEN_SHAPE/term`; this is how the
    /// provider table tells the two apart.
    pub fn is_filling_target(&self, target: &Target, family: Family) -> bool {
        self.filling
            .iter()
            .any(|key| &key.target == target && key.family == family)
    }

    pub fn open(&mut self, key: MemoKey) {
        self.filling.push(key);
    }

    pub fn close(&mut self) {
        self.filling.pop();
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
