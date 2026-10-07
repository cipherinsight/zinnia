//! Stale arrays: storages that no longer hold what NumPy would show.
//!
//! Where NumPy shares one array between two names (a view, or a name bound
//! to one of two arrays depending on a runtime condition) and Zinnia keeps
//! two storages, the IR generator *links* them (`ir_gen/detached_copies.rs`,
//! [`IRBuilder::link_storages`]). A write to either one then leaves the
//! other one *stale*: its cells miss the write. Two rules make that sound
//! whatever the syntax of the program, each enforced in one place:
//!
//! * every write to a storage goes through the store, which marks every
//!   storage linked to it stale (`note_storage_written`; an update that
//!   rebinds a name to a new storage notes the old one as replaced);
//! * every read of a storage's cells goes through the store, which rejects
//!   reading a stale storage with a compile error naming the reason (a name,
//!   a method call, an element of a nested container, an output).
//!
//! A merge (`helpers::value_ops::select_value`) reads its operands without
//! the check and carries their staleness over to its result, so a merged
//! value that is never read costs nothing and one that is read is rejected.
//!
//! **Paths.** A storage is stale only on the paths where the write ran:
//! staleness is a set of *paths*, each the conjunction of the literals of
//! the write's execution predicate (a literal is a Boolean statement or its
//! negation; the empty conjunction is every path). A read is rejected when
//! one of its storage's stale paths can hold together with the read's own
//! execution predicate, that is when the two conjunctions hold no literal
//! and its negation. A merge `select(c, a, b)` carries `a`'s paths
//! conjoined with `c` and `b`'s with `not c`, so a branch merge that takes
//! a stale array only on paths where it is not stale stays readable. The
//! test is syntactic and errs towards rejecting: two conjunctions it does
//! not prove disjoint are treated as overlapping.

use std::collections::BTreeSet;

use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::IRBuilder;

/// Why a copy Zinnia made of what NumPy returns as a view is stale.
pub const COPIED_VIEW: &str = "it is linked by NumPy view semantics to an array that was modified \
     afterwards, but Zinnia had to copy it (the view's position depends on a private index, it is \
     a reversed or runtime-bounded slice, a reshape of a non-contiguous array, or the result of a \
     function NumPy returns a view from, such as `np.flip` or `np.rot90`), so it would not show \
     the modification. Index the original array directly instead of keeping the view.";

/// A Boolean statement, or its negation (`false`).
type Literal = (StmtId, bool);

/// A set of paths: those where a conjunction of literals holds, or those
/// where it does not.
#[derive(Clone, Debug)]
enum Paths {
    Where(Vec<Literal>),
    WhereNot(Vec<Literal>),
}

impl Paths {
    /// The paths of `path` that are also in `self`, as a conjunction, or
    /// `None` when there are none. `path` and `not (l1 and l2 ...)` is not a
    /// conjunction unless the negated one has a single literal; it is
    /// over-approximated by `path`, unless `path` implies the negated
    /// conjunction (then there are none).
    fn within(&self, path: &[Literal]) -> Option<Vec<Literal>> {
        match self {
            Paths::Where(conjunction) => conjoin(path, conjunction),
            Paths::WhereNot(conjunction) => match conjunction.as_slice() {
                [] => None,
                [(s, positive)] => conjoin(path, &[(*s, !positive)]),
                _ if conjunction.iter().all(|l| path.binary_search(l).is_ok()) => None,
                _ => Some(path.to_vec()),
            },
        }
    }
}

/// Two storages Python would share and Zinnia keeps apart: `copy` holds
/// what NumPy would make the same memory as `base` on the paths `when`. A
/// view's copy is the base everywhere; a merge `select(c, a, b)` is `a`
/// where `c` holds and `b` where it does not.
#[derive(Clone, Debug)]
pub(crate) struct Link {
    copy: u32,
    base: u32,
    why: &'static str,
    when: Paths,
}

/// Where a storage is stale: the reason, and the paths (conjunctions of
/// literals, sorted; empty: every path).
#[derive(Clone, Debug, Default)]
pub(crate) struct Staleness {
    why: Option<&'static str>,
    paths: Vec<Vec<Literal>>,
}

impl Staleness {
    pub(super) fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    fn add(&mut self, why: &'static str, path: Vec<Literal>) {
        self.why.get_or_insert(why);
        // A path that a recorded path contains adds nothing.
        if self.paths.iter().any(|p| p.iter().all(|l| path.binary_search(l).is_ok())) {
            return;
        }
        self.paths.retain(|p| !path.iter().all(|l| p.binary_search(l).is_ok()));
        self.paths.push(path);
    }
}

/// Whether two conjunctions can hold together (no literal and its negation).
fn compatible(a: &[Literal], b: &[Literal]) -> bool {
    !a.iter().any(|&(s, positive)| b.binary_search(&(s, !positive)).is_ok())
}

/// The conjunction of two conjunctions, or `None` when it cannot hold.
fn conjoin(a: &[Literal], b: &[Literal]) -> Option<Vec<Literal>> {
    if !compatible(a, b) {
        return None;
    }
    let mut out: Vec<Literal> = a.iter().chain(b).copied().collect::<BTreeSet<_>>().into_iter().collect();
    out.dedup();
    Some(out)
}

impl IRBuilder {
    /// Record that `copy` holds what NumPy would make the same memory as
    /// `base`, for the reason `why`: from then on a write to either one
    /// makes the other one stale. Links are transitive.
    pub fn link_storages(&mut self, copy: u32, base: u32, why: &'static str) {
        self.add_link(copy, base, why, Paths::Where(Vec::new()));
    }

    /// Record that `merged`, a merge `select(cond, a, b)` of arrays a name
    /// was bound to on different paths, is `source` where `cond` holds
    /// (`taken`: `source` is `a`) or where it does not (`source` is `b`). A
    /// write to either one then makes the other stale on those paths only.
    pub fn link_merged_storage(&mut self, merged: u32, source: u32, why: &'static str, cond: &Value, taken: bool) {
        let Some(literals) = self.path_of(Some(cond)) else {
            // `cond` is the constant false: `merged` is `b` everywhere.
            if !taken {
                self.add_link(merged, source, why, Paths::Where(Vec::new()));
            }
            return;
        };
        let when = if taken { Paths::Where(literals) } else { Paths::WhereNot(literals) };
        self.add_link(merged, source, why, when);
    }

    fn add_link(&mut self, copy: u32, base: u32, why: &'static str, when: Paths) {
        let (copy, base) = (self.storage_owner(copy), self.storage_owner(base));
        if copy != base {
            self.store.links.push(Link { copy, base, why, when });
        }
    }

    /// Record that the array of `storage` was updated by an operation that
    /// built the new contents in another storage and rebinds the name to it
    /// (no store write to `storage` happens, but NumPy changed that array):
    /// the storages linked to it are stale, as after a write.
    pub fn note_storage_replaced(&mut self, storage: u32) {
        self.propagate_write_to_links(storage);
    }

    /// The array of `old` now lives in `new` (an update that builds the new
    /// contents in another storage and rebinds the name to it): `new` takes
    /// over `old`'s links, so a later write through a linked copy makes it
    /// stale, as it would `old`.
    pub fn move_links(&mut self, old: u32, new: u32) {
        let (old, new) = (self.storage_owner(old), self.storage_owner(new));
        if old == new {
            return;
        }
        let moved: Vec<Link> = self
            .store
            .links
            .iter()
            .filter(|link| link.copy == old || link.base == old)
            .map(|link| {
                let mut link = link.clone();
                if link.copy == old {
                    link.copy = new;
                } else {
                    link.base = new;
                }
                link
            })
            .collect();
        self.store.links.extend(moved);
    }

    /// Why some array in `value` (an array, or a list or tuple holding
    /// arrays at any depth) is stale on a path the current code can run on,
    /// if one is.
    pub fn stale_reason_in(&mut self, value: &Value) -> Option<&'static str> {
        let here = self.current_path()?;
        self.stalenesses_in(value).into_iter().find_map(|stale| {
            stale.paths.iter().any(|p| compatible(p, &here)).then_some(stale.why).flatten()
        })
    }

    /// Make every array in `value` (at any depth) read-only: NumPy returns it
    /// that way (`why`), and a write to it raises `ValueError`.
    pub fn mark_read_only_in(&mut self, value: &Value, why: &'static str) {
        match value {
            Value::StaticArray { segment_id, .. } => {
                let owner = self.storage_owner(*segment_id);
                self.storage_mut(owner).read_only = Some(why);
            }
            Value::DynamicNDArray(data) => {
                let owner = self.storage_owner(data.segment_id);
                self.storage_mut(owner).read_only = Some(why);
            }
            Value::List(data) | Value::Tuple(data) => {
                for v in &data.values {
                    self.mark_read_only_in(v, why);
                }
            }
            _ => {}
        }
    }

    /// Run `f` with reads of stale storages allowed: for a computation that
    /// carries staleness over to its result (`select_value`).
    pub fn with_stale_reads<T>(&mut self, f: impl FnOnce(&mut IRBuilder) -> T) -> T {
        self.store.stale_reads_allowed += 1;
        let out = f(self);
        self.store.stale_reads_allowed -= 1;
        out
    }

    /// Whether some array in `value` is stale on some path.
    pub fn has_stale(&self, value: &Value) -> bool {
        !self.stalenesses_in(value).is_empty()
    }

    /// Mark the arrays of `out`, a merge `select(cond, tv, fv)`, stale where
    /// the operand it takes there is: `tv`'s stale paths under `cond`, and
    /// `fv`'s under `not cond`.
    pub fn carry_merge_staleness(&mut self, cond: &Value, tv: &Value, fv: &Value, out: &Value) {
        let Some(taken) = self.path_of(Some(cond)) else { return };
        let (where_taken, where_not) = (Paths::Where(taken.clone()), Paths::WhereNot(taken));
        let mut carried: Vec<(&'static str, Vec<Literal>)> = Vec::new();
        for (side, paths) in [(tv, &where_taken), (fv, &where_not)] {
            for stale in self.stalenesses_in(side) {
                let why = stale.why.expect("a stale storage has a reason");
                carried.extend(stale.paths.iter().filter_map(|p| paths.within(p)).map(|p| (why, p)));
            }
        }
        for (why, path) in carried {
            self.mark_stale_in(out, why, &path);
        }
    }

    // ── Internals (also used by the store) ────────────────────────────

    /// The storage that identifies an array: a complex array's real part.
    pub(super) fn storage_owner(&self, storage: u32) -> u32 {
        self.storage(storage).real.unwrap_or(storage)
    }

    /// Reject reading the cells of `storage` where it is stale on a path the
    /// current code can run on.
    pub(super) fn check_readable(&mut self, storage: u32) {
        if self.store.stale_reads_allowed > 0 {
            return;
        }
        let owner = self.storage_owner(storage);
        if self.storage(owner).stale.paths.is_empty() {
            return;
        }
        let Some(here) = self.current_path() else { return };
        let stale = &self.storage(owner).stale;
        if stale.paths.iter().any(|p| compatible(p, &here)) {
            let why = stale.why.expect("a stale storage has a reason");
            panic!("Reading this array is not supported here: {why}");
        }
    }

    /// A write to `storage` (under the current execution predicate) makes
    /// every storage linked to it stale on the write's path, conjoined with
    /// the paths on which each link along the way shares the memory.
    pub(super) fn propagate_write_to_links(&mut self, storage: u32) {
        if self.store.links.is_empty() {
            return;
        }
        let Some(path) = self.current_path() else { return };
        let start = self.storage_owner(storage);
        // (storage, path) pairs: a storage reached on two paths is stale on
        // both. Paths only grow, so this terminates.
        let mut seen = BTreeSet::from([(start, path.clone())]);
        let mut frontier = vec![(start, path)];
        let mut reached: Vec<(u32, &'static str, Vec<Literal>)> = Vec::new();
        while let Some((current, path)) = frontier.pop() {
            for link in &self.store.links {
                let other = if link.copy == current {
                    link.base
                } else if link.base == current {
                    link.copy
                } else {
                    continue;
                };
                if other == start {
                    continue;
                }
                let Some(shared) = link.when.within(&path) else { continue };
                if seen.insert((other, shared.clone())) {
                    reached.push((other, link.why, shared.clone()));
                    frontier.push((other, shared));
                }
            }
        }
        for (other, why, path) in reached {
            self.storage_mut(other).stale.add(why, path);
        }
    }

    /// The stalenesses of the arrays in `value` (at any depth).
    fn stalenesses_in(&self, value: &Value) -> Vec<Staleness> {
        let of = |storage: u32| {
            let stale = &self.storage(self.storage_owner(storage)).stale;
            (!stale.paths.is_empty()).then(|| stale.clone())
        };
        match value {
            Value::StaticArray { segment_id, imag_segment_id, .. } => {
                of(*segment_id).into_iter().chain(imag_segment_id.and_then(of)).collect()
            }
            Value::DynamicNDArray(data) => of(data.segment_id).into_iter().collect(),
            Value::List(data) | Value::Tuple(data) => data.values.iter().flat_map(|v| self.stalenesses_in(v)).collect(),
            _ => Vec::new(),
        }
    }

    /// Mark every array in `value` (at any depth) stale on `path`.
    fn mark_stale_in(&mut self, value: &Value, why: &'static str, path: &[Literal]) {
        match value {
            Value::StaticArray { segment_id, .. } => {
                let owner = self.storage_owner(*segment_id);
                self.storage_mut(owner).stale.add(why, path.to_vec());
            }
            Value::DynamicNDArray(data) => {
                let owner = self.storage_owner(data.segment_id);
                self.storage_mut(owner).stale.add(why, path.to_vec());
            }
            Value::List(data) | Value::Tuple(data) => {
                for v in &data.values {
                    self.mark_stale_in(v, why, path);
                }
            }
            _ => {}
        }
    }

    /// The conjunction of the current execution predicate's literals
    /// (`None`: the current code never runs).
    fn current_path(&mut self) -> Option<Vec<Literal>> {
        let p = self.write_predicate().cloned();
        self.path_of(p.as_ref())
    }

    /// The literals of the predicate `p` (`None` for no predicate: every
    /// path; the result `None` when `p` is the constant false).
    fn path_of(&mut self, p: Option<&Value>) -> Option<Vec<Literal>> {
        let Some(p) = p else { return Some(Vec::new()) };
        match p.bool_val().or_else(|| p.int_val().map(|v| v != 0)) {
            Some(true) => return Some(Vec::new()),
            Some(false) => return None,
            None => {}
        }
        let atoms = self.predicate_atoms(p);
        let mut literals: Vec<Literal> = atoms.iter().map(|&a| self.literal(a)).collect();
        literals.sort_unstable();
        literals.dedup();
        compatible(&literals, &literals).then_some(literals)
    }

    /// A Boolean statement as a literal: `not x` (any depth of negation) is
    /// `x` negated.
    fn literal(&self, mut statement: StmtId) -> Literal {
        let mut positive = true;
        while let Some(stmt) = self.stmts.get(statement as usize) {
            match (&stmt.ir, stmt.arguments.as_slice()) {
                (IR::LogicalNot, &[x]) => {
                    statement = x;
                    positive = !positive;
                }
                _ => break,
            }
        }
        (statement, positive)
    }
}
