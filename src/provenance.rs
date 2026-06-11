//! Structural provenance: ONE table with written semantics.
//!
//! The 2026-06 benchmark campaign repeatedly hit the same meta-problem:
//! *a lowering knows something structural at emit time; a later consumer
//! needs it; no general channel exists*. Each instance grew an ad-hoc
//! cache (`slice_views`, `argwhere_keep_indicators`, the halo2/Noir
//! affine+interval+indicator caches), and the worst bugs of the campaign
//! were provenance-semantics bugs. This module is the consolidation
//! target: structural records with explicit semantics, owned by the
//! `IRBuilder` during IR generation and carried onto the `IRGraph` for
//! backend consumption.
//!
//! The complementary LOGICAL layer is `FactStack`
//! (`src/optim/predicates/facts.rs`): scoped `ContractTerm` facts with
//! Z3-backed `prove()`. Facts answer "is this proposition true"; this
//! table answers "how was this value/segment built". Keep both; never
//! duplicate one into the other.
//!
//! # Invariants (the campaign's hard-won rules — keep them true)
//!
//! 1. **Layout**: a padded buffer requires `runtime_shape[i] <=
//!    logical_shape[i]` on every axis. An axis whose bound exceeds its
//!    logical size is a runtime-reshape skeleton axis ⇒ the buffer is
//!    compact and addressing uses runtime strides.
//!    (`src/ops/dyn_ndarray/aggregation.rs` encodes this today.)
//! 2. **Structural vs witness values**: any value cache that sees witness
//!    data must never drive emission decisions. Decision sites use
//!    structural-only accessors (the Noir emitter's `s_int_of` /
//!    `literal_cells` pattern). This applies to every backend.
//! 3. **Runtime-axis bound**: each runtime axis of a reshape is bounded
//!    by `total / product(static dims)`; exceeding it is unsatisfiable
//!    via the divisibility assert.
//! 4. **Invalidation**: any structural record derived from segment
//!    contents is keyed by (segment, write_version); writes bump the
//!    version unconditionally. A record whose stored version is older
//!    than the segment's current version is dead.
//! 5. **Membership soundness**: a shared-indicator gather must
//!    one-hot-assert its enumerated value set once per (root, value-set);
//!    silently gathering init/zero on an out-of-set witness is the bug
//!    class that motivated this.
//!
//! # Identity rule
//!
//! Graph-resident records reference values by `ValueId` ONLY — `ValueId`s
//! are stable across optimization passes ("arg_values is in ValueId
//! space"), while `StmtId`s renumber on every rebuilding pass. Consumers
//! resolve a `ValueId` to its producing statement through
//! `IRGraph::value_to_stmt` at consumption time. The one exception is
//! [`SliceView`], which carries a full `Value` for its builder-phase
//! consumers (it predates this module and is consumed before any pass
//! runs); graph-phase consumers must re-resolve through its `value_id()`.
//!
//! Passes that rewrite value identity call [`ProvenanceTable::remap_value`]
//! for every replaced id; passes that delete values call
//! [`ProvenanceTable::retain_values`] so records are dropped, never left
//! dangling. [`ProvenanceTable::debug_consistency_check`] validates the
//! result (enabled with `ZINNIA_PROVENANCE_CHECK=1`, and always in tests).

use std::collections::HashMap;

use crate::types::{Value, ValueId};

/// Affine view of a scalar: `value = Σ terms[i].1 * vid_i + offset`,
/// with at most two dynamic roots. Backend-independent replacement for
/// the halo2 synthesizer's `AffineLin` and the Noir emitter's `AffLin`
/// (consumed there from Step 2 of the consolidation plan).
#[derive(Clone, Debug)]
pub struct AffineView {
    pub terms: Vec<(ValueId, i64)>,
    pub offset: i64,
}

impl AffineView {
    pub fn constant(v: i64) -> Self {
        AffineView {
            terms: Vec::new(),
            offset: v,
        }
    }

    pub fn root(vid: ValueId) -> Self {
        AffineView {
            terms: vec![(vid, 1)],
            offset: 0,
        }
    }

    /// `a + sign*b`, refusing results with more than two roots or
    /// overflowing coefficients (mirrors the backends' AffineLin/AffLin
    /// combine; refusal means "no affine record", never wrong math).
    pub fn combine(a: &AffineView, b: &AffineView, sign: i64) -> Option<AffineView> {
        let offset = (a.offset as i128) + (sign as i128) * (b.offset as i128);
        let mut terms = a.terms.clone();
        for (root, scale) in &b.terms {
            let s = (sign as i128) * (*scale as i128);
            if let Some(existing) = terms.iter_mut().find(|(r, _)| r == root) {
                let merged = (existing.1 as i128) + s;
                if merged.abs() > i64::MAX as i128 {
                    return None;
                }
                existing.1 = merged as i64;
            } else {
                if s.abs() > i64::MAX as i128 {
                    return None;
                }
                terms.push((*root, s as i64));
            }
        }
        terms.retain(|(_, s)| *s != 0);
        if terms.len() > 2 || offset.abs() > i64::MAX as i128 {
            return None;
        }
        Some(AffineView {
            terms,
            offset: offset as i64,
        })
    }

    /// `self * k`, with the same refusal rules.
    pub fn scaled(&self, k: i64) -> Option<AffineView> {
        let offset = (self.offset as i128) * (k as i128);
        if offset.abs() > i64::MAX as i128 {
            return None;
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for (root, scale) in &self.terms {
            let s = (*scale as i128) * (k as i128);
            if s.abs() > i64::MAX as i128 {
                return None;
            }
            if s != 0 {
                terms.push((*root, s as i64));
            }
        }
        Some(AffineView {
            terms,
            offset: offset as i64,
        })
    }
}

/// Inclusive value bounds. The structural counterpart of the backends'
/// `value_interval` / `interval_of` caches.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

impl Interval {
    pub const BIT: Interval = Interval { lo: 0, hi: 1 };

    pub fn point(v: i64) -> Self {
        Interval { lo: v, hi: v }
    }

    pub fn union(a: Interval, b: Interval) -> Interval {
        Interval {
            lo: a.lo.min(b.lo),
            hi: a.hi.max(b.hi),
        }
    }

    /// `a + sign*b` with overflow refusal.
    pub fn add(a: Interval, b: Interval, sign: i64) -> Option<Interval> {
        let (blo, bhi) = if sign >= 0 { (b.lo, b.hi) } else { (-b.hi, -b.lo) };
        let lo = (a.lo as i128) + (blo as i128);
        let hi = (a.hi as i128) + (bhi as i128);
        if lo.abs() > i64::MAX as i128 || hi.abs() > i64::MAX as i128 {
            return None;
        }
        Some(Interval {
            lo: lo as i64,
            hi: hi as i64,
        })
    }

    /// `self * k` (k may be negative) with overflow refusal.
    pub fn scaled(&self, k: i64) -> Option<Interval> {
        let a = (self.lo as i128) * (k as i128);
        let b = (self.hi as i128) * (k as i128);
        if a.abs() > i64::MAX as i128 || b.abs() > i64::MAX as i128 {
            return None;
        }
        Some(Interval {
            lo: a.min(b) as i64,
            hi: a.max(b) as i64,
        })
    }

    pub fn intersect(a: Interval, b: Interval) -> Interval {
        Interval {
            lo: a.lo.max(b.lo),
            hi: a.hi.min(b.hi),
        }
    }
}

/// How a dyn-array buffer is laid out (invariant #1). Every dyn-array
/// producer must state which layout it built; consumers must not guess
/// from `runtime_strides` (the confusion that produced the 0415/0416/0418
/// wrong-output binding bug).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    /// Content sits at the logical positions of the envelope rectangle;
    /// `runtime_shape[i] <= logical_shape[i]` on every axis; addressing
    /// uses the (static) logical strides.
    Padded,
    /// Compact buffer in runtime C-order; addressing uses the (dynamic)
    /// runtime strides; `prod(logical_shape) == total_bound` is the
    /// padding-rectangle invariant.
    Compact,
}

/// Compaction provenance: this segment's payload is `keep`-filtered
/// content packed to the front (zero-padded tail), with `count` kept
/// elements. `prefix[i]` = number of kept elements among positions
/// `0..=i` (the scatter/gather alignment used by mask writes). `sorted`
/// records whether the packed values are non-decreasing (true for
/// `flatnonzero`/`argwhere`-style index outputs), which unlocks
/// first/last-set-bit consumers for `min()`/`max()` (plan Step 3).
#[derive(Clone, Debug)]
pub struct CompactionInfo {
    pub keep: Vec<ValueId>,
    pub prefix: Vec<ValueId>,
    pub count: ValueId,
    pub sorted: bool,
}

/// Affine slice provenance: element `j` of the annotated array equals
/// `base[base_offset + (start + j*step) * stride]` (axis-domain map into
/// a flat base segment) for every in-range `j < runtime_length`. `start`
/// is an IR value (possibly dynamic); `step`/geometry are static.
///
/// Moved verbatim from `IRBuilder` as the consolidation pilot; see the
/// module-level identity rule about its `Value` field.
#[derive(Clone, Debug)]
pub struct SliceView {
    pub base_segment: u32,
    pub axis_len: usize,
    pub stride: usize,
    pub base_offset: usize,
    pub start: Value,
    pub step: i64,
}

/// The one structural-provenance table. Owned by `IRBuilder` during IR
/// generation; transferred onto `IRGraph` at export and carried through
/// the optimization pipeline (with remap/retain hooks) so backends can
/// consume it (plan Step 2).
#[derive(Default, Clone, Debug)]
pub struct ProvenanceTable {
    /// Scalar affine views, keyed by the value they describe.
    affine: HashMap<ValueId, AffineView>,
    /// Scalar bounds, keyed by the value they bound.
    interval: HashMap<ValueId, Interval>,
    /// Buffer layout per segment.
    layout: HashMap<u32, LayoutKind>,
    /// Compaction provenance per segment.
    compaction: HashMap<u32, CompactionInfo>,
    /// Slice provenance for materialised dynamic 1-D slices, keyed by the
    /// OUTPUT segment. Invalidated when either the view's own segment or
    /// its base segment is written (invariant #4; eager-drop form).
    slice_views: HashMap<u32, SliceView>,
    /// Monotone per-segment write counters (invariant #4). Bumped
    /// unconditionally by `ir_write_memory`; segment-derived records that
    /// outlive the eager-drop policy must carry the version they were
    /// built at and compare against this.
    write_version: HashMap<u32, u64>,
    /// Per-segment value hull: union of the init value and every written
    /// value's interval. `None` entry = poisoned (some write had no
    /// interval). Feeds `ReadMemory` output intervals (the Noir emitter's
    /// `segment_iv` semantics).
    segment_hull: HashMap<u32, Option<Interval>>,
}

impl ProvenanceTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// True when no records are held (write-version counters don't
    /// count — they're bookkeeping, not records).
    pub fn is_empty(&self) -> bool {
        self.affine.is_empty()
            && self.interval.is_empty()
            && self.layout.is_empty()
            && self.compaction.is_empty()
            && self.slice_views.is_empty()
    }

    // ── Write tracking (invariant #4) ────────────────────────────────

    /// Record a write to `segment`: bumps its version and drops every
    /// segment-derived record invalidated by the write. Mirrors (and now
    /// centralizes) the eager invalidation `ir_write_memory` used to do
    /// inline for slice views.
    pub fn on_segment_write(&mut self, segment: u32) {
        *self.write_version.entry(segment).or_insert(0) += 1;
        self.slice_views
            .retain(|seg, view| *seg != segment && view.base_segment != segment);
        self.compaction.remove(&segment);
    }

    pub fn write_version(&self, segment: u32) -> u64 {
        self.write_version.get(&segment).copied().unwrap_or(0)
    }

    // ── Segment value hulls ──────────────────────────────────────────

    pub fn init_segment_hull(&mut self, segment: u32, init: i64) {
        self.segment_hull.insert(segment, Some(Interval::point(init)));
    }

    /// Union a written value's interval into the segment hull; `None`
    /// poisons the hull (unknown write).
    pub fn widen_segment_hull(&mut self, segment: u32, iv: Option<Interval>) {
        let entry = self.segment_hull.entry(segment).or_insert(None);
        *entry = match (entry.take(), iv) {
            (Some(a), Some(b)) => Some(Interval::union(a, b)),
            _ => None,
        };
    }

    pub fn segment_hull(&self, segment: u32) -> Option<Interval> {
        self.segment_hull.get(&segment).copied().flatten()
    }

    // ── Record accessors ─────────────────────────────────────────────

    pub fn slice_view(&self, segment: u32) -> Option<&SliceView> {
        self.slice_views.get(&segment)
    }

    pub fn set_slice_view(&mut self, segment: u32, view: SliceView) {
        self.slice_views.insert(segment, view);
    }

    pub fn affine_of(&self, vid: ValueId) -> Option<&AffineView> {
        self.affine.get(&vid)
    }

    pub fn set_affine(&mut self, vid: ValueId, view: AffineView) {
        self.affine.insert(vid, view);
    }

    pub fn interval_of(&self, vid: ValueId) -> Option<Interval> {
        self.interval.get(&vid).copied()
    }

    pub fn set_interval(&mut self, vid: ValueId, iv: Interval) {
        self.interval.insert(vid, iv);
    }

    pub fn layout_of(&self, segment: u32) -> Option<LayoutKind> {
        self.layout.get(&segment).copied()
    }

    pub fn set_layout(&mut self, segment: u32, kind: LayoutKind) {
        self.layout.insert(segment, kind);
    }

    pub fn compaction_of(&self, segment: u32) -> Option<&CompactionInfo> {
        self.compaction.get(&segment)
    }

    pub fn set_compaction(&mut self, segment: u32, info: CompactionInfo) {
        self.compaction.insert(segment, info);
    }

    // ── Pass hooks ───────────────────────────────────────────────────

    /// A pass replaced every use of `old` with `new` (duplicate-code
    /// elimination): re-key records and rewrite term lists.
    pub fn remap_value(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        if let Some(v) = self.affine.remove(&old) {
            self.affine.entry(new).or_insert(v);
        }
        if let Some(v) = self.interval.remove(&old) {
            self.interval.entry(new).or_insert(v);
        }
        for view in self.affine.values_mut() {
            for (vid, _) in view.terms.iter_mut() {
                if *vid == old {
                    *vid = new;
                }
            }
        }
        for info in self.compaction.values_mut() {
            for vid in info.keep.iter_mut().chain(info.prefix.iter_mut()) {
                if *vid == old {
                    *vid = new;
                }
            }
            if info.count == old {
                info.count = new;
            }
        }
    }

    /// A pass deleted values: drop every record that references a dead
    /// `ValueId` (records must never dangle — invariant of the module).
    pub fn retain_values(&mut self, live: &dyn Fn(ValueId) -> bool) {
        self.affine
            .retain(|vid, view| live(*vid) && view.terms.iter().all(|(t, _)| live(*t)));
        self.interval.retain(|vid, _| live(*vid));
        self.compaction.retain(|_, info| {
            live(info.count)
                && info.keep.iter().all(|v| live(*v))
                && info.prefix.iter().all(|v| live(*v))
        });
        self.slice_views.retain(|_, view| {
            view.start.value_id().map(|v| live(v)).unwrap_or(true)
        });
    }

    /// Validate that every `ValueId` referenced by a record resolves in
    /// the graph (run at the end of the compile pipeline when
    /// `ZINNIA_PROVENANCE_CHECK=1`; always run by tests). Returns the
    /// offending description instead of panicking so callers control
    /// severity.
    pub fn debug_consistency_check(
        &self,
        value_exists: &dyn Fn(ValueId) -> bool,
    ) -> Result<(), String> {
        for (vid, view) in &self.affine {
            if !value_exists(*vid) {
                return Err(format!("affine record keyed by dead value {:?}", vid));
            }
            for (t, _) in &view.terms {
                if !value_exists(*t) {
                    return Err(format!("affine term references dead value {:?}", t));
                }
            }
        }
        for vid in self.interval.keys() {
            if !value_exists(*vid) {
                return Err(format!("interval record keyed by dead value {:?}", vid));
            }
        }
        for (seg, info) in &self.compaction {
            for v in info.keep.iter().chain(info.prefix.iter()) {
                if !value_exists(*v) {
                    return Err(format!(
                        "compaction record for segment {} references dead value {:?}",
                        seg, v
                    ));
                }
            }
            if !value_exists(info.count) {
                return Err(format!(
                    "compaction record for segment {} has dead count {:?}",
                    seg, info.count
                ));
            }
        }
        for (seg, view) in &self.slice_views {
            if let Some(vid) = view.start.value_id() {
                if !value_exists(vid) {
                    return Err(format!(
                        "slice view for segment {} has dead start value {:?}",
                        seg, vid
                    ));
                }
            }
        }
        Ok(())
    }
}
