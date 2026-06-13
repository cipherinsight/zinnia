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

use crate::types::{StmtId, Value, ValueId};

/// Affine view of a scalar: `value = Σ terms[i].1 * vid_i + offset`,
/// with at most two dynamic roots. Backend-independent replacement for
/// the halo2 synthesizer's `AffineLin` and the Noir emitter's `AffLin`
/// (consumed there from Step 2 of the consolidation plan).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// Compaction provenance: this segment lane's payload is `keep`-filtered
/// content packed to the front (zero-padded tail), with `count` kept
/// elements. `source[i]` is the pre-compaction value at source position
/// `i` (for index-style outputs these are the coordinate constants);
/// `prefix[i]` = number of kept elements among positions `0..=i` (the
/// scatter/gather alignment used by mask writes). `sorted` records
/// whether the packed values are non-decreasing in source order (true
/// for `flatnonzero`/`argwhere`-style index outputs), which unlocks
/// first/last-set-bit scan consumers for `min()`/`max()`: O(N) selects
/// over (keep, source) instead of materialise + masked reduce.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompactionInfo {
    pub keep: Vec<ValueId>,
    pub source: Vec<ValueId>,
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

/// Mask-read provenance (builder-phase only, like [`SliceView`]: carries
/// `Value`s, never serialized): the annotated segment holds `arr[mask]`
/// — the elements of `sources` whose `mask_cells` bit is true, packed to
/// the front in source order, default-padded. Both vectors are indexed
/// by SOURCE position and snapshot read-time SSA values, so the record
/// stays correct under later writes to the source array (numpy evaluates
/// the RHS of `a[mask] op= v` before the store).
///
/// Deposited by `dynamic_mask_read` (whole-element masks only),
/// re-deposited lane-transformed by `dyn_scalar_binary_op`, and consumed
/// by `try_apply_boolean_mask_write` to cancel the pack/unpack round
/// trip of `a[mask] op= v` into per-element selects (O(N) instead of
/// the O(N^2) shift network + prefix one-hot). Eagerly dropped when the
/// segment is written (`on_segment_write`).
#[derive(Clone, Debug)]
pub struct MaskReadView {
    pub mask_cells: Vec<Value>,
    pub sources: Vec<Value>,
    /// Producing stmt of the packed array's runtime length (the true
    /// count). Consumers must require the consumed dyn value's
    /// `runtime_length` to carry this stmt — a sliced/reshaped re-view
    /// of the same segment mints a different length and must NOT be
    /// consumed positionally.
    pub count_stmt: Option<StmtId>,
}

/// Structural clamp: the annotated value is `min(source, bound)` (or
/// `max(source, bound)` when `is_min` is false) with a compile-time
/// constant `bound`. Deposited by the `SelectI` min/max recognizer when
/// one branch is constant. Lets backend equality indicators on clamped
/// gather addresses reduce to the un-clamped root's indicator family
/// (`[min(x,K) == v]` is `[x == v]` for `v < K` and `[x >= K]` at the
/// bound), so per-row dynamic gathers share one one-hot per root.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ClampView {
    pub source: ValueId,
    pub bound: i64,
    pub is_min: bool,
}

/// Block-confined view: for the annotated DynamicNDArray VALUE, every
/// active (within-runtime-shape) element's storage position lies inside
/// the static block selected by its leading `prefix_rank` coordinates —
/// `[linear(prefix coords) * block, … + block)`. Deposited by the
/// padded-prefix runtime-reshape path (a dynamic-bounds slice of a
/// static array reshaped without repacking); the producer's
/// `prod(runtime_shape) == runtime_length` assert plus the slice
/// extent's `<= envelope dim` construction guarantee the confinement.
/// Consumed by axis reductions to scan one block per output instead of
/// the whole buffer, with window masks shared across prefix coords.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct BlockView {
    pub prefix_rank: usize,
    pub block: usize,
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
    /// Compaction provenance per (segment, lane). 1-D compactions use
    /// lane 0; interleaved outputs (argwhere's (count, rank) rows) hold
    /// one record per coordinate lane.
    compaction: HashMap<(u32, usize), CompactionInfo>,
    /// Slice provenance for materialised dynamic 1-D slices, keyed by the
    /// OUTPUT segment. Invalidated when either the view's own segment or
    /// its base segment is written (invariant #4; eager-drop form).
    slice_views: HashMap<u32, SliceView>,
    /// Mask-read provenance keyed by the OUTPUT segment (builder-phase
    /// only, like `slice_views`). Eagerly dropped when the segment is
    /// written.
    mask_read_views: HashMap<u32, MaskReadView>,
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
    /// Runtime extents of a PADDED slice materialization: the segment's
    /// content occupies the `[0, extent_i)` prefix of axis `i` of its
    /// envelope rectangle (one ValueId per axis; constants for static
    /// axes). Deposited by dynamic-bounds slices of static arrays;
    /// consumed when the value is promoted to a dyn array or bound as a
    /// circuit output, so the runtime shape is pinned by the contract
    /// instead of leaking the envelope skeleton (the 0507/0509/0510
    /// under-binding).
    runtime_extents: HashMap<u32, Vec<ValueId>>,
    /// Block-confined dyn-array views, keyed by the array VALUE (not the
    /// segment — confinement is a property of the view's strides).
    /// Consumed at IR-build time by axis reductions; carried in the
    /// table for the uniform remap/retain/consistency lifecycle.
    block_views: HashMap<ValueId, BlockView>,
    /// Structural min/max-against-constant views, keyed by the clamped
    /// value.
    clamps: HashMap<ValueId, ClampView>,
}


/// JSON-friendly wire form of the graph-phase records, carried inside
/// proof artifacts whose verify path re-synthesizes the circuit from
/// serialized IR: backend emission consumes the table (Step 2), so a
/// roundtrip that dropped it produced a DIFFERENT (bigger) circuit at
/// verify time ("break points don't match" on the provenance-heavy
/// bbox kernels).
///
/// Values are referenced by STMT id, not `ValueId`:
/// `IRStatement::import_from` mints fresh `ValueId`s, so vid-keyed
/// records would dangle after a JSON roundtrip, while stmt ids survive
/// it verbatim. Conversion happens against a graph's vid↔stmt maps;
/// records whose vid no longer resolves to a statement are dropped
/// (they could never influence synthesis). Entry vectors are key-sorted
/// so artifact bytes stay deterministic. `slice_views` is
/// builder-phase-only (carries a full `Value`) and is deliberately not
/// part of the wire form.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ProvenanceWire {
    affine: Vec<(StmtId, Vec<(StmtId, i64)>, i64)>,
    interval: Vec<(StmtId, Interval)>,
    layout: Vec<(u32, LayoutKind)>,
    compaction: Vec<(u32, usize, CompactionWire)>,
    write_version: Vec<(u32, u64)>,
    segment_hull: Vec<(u32, Option<Interval>)>,
    runtime_extents: Vec<(u32, Vec<StmtId>)>,
    clamps: Vec<(StmtId, StmtId, i64, bool)>,
    block_views: Vec<(StmtId, BlockView)>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompactionWire {
    pub keep: Vec<StmtId>,
    pub source: Vec<StmtId>,
    pub prefix: Vec<StmtId>,
    pub count: StmtId,
    pub sorted: bool,
}

impl ProvenanceTable {
    /// Serialize against `stmt_of` (vid → producing stmt). Records with
    /// any unresolvable vid are dropped.
    pub fn to_wire(&self, stmt_of: &dyn Fn(ValueId) -> Option<StmtId>) -> ProvenanceWire {
        let map_all = |vids: &[ValueId]| -> Option<Vec<StmtId>> {
            vids.iter().map(|v| stmt_of(*v)).collect()
        };
        let mut affine: Vec<(StmtId, Vec<(StmtId, i64)>, i64)> = self
            .affine
            .iter()
            .filter_map(|(vid, view)| {
                let key = stmt_of(*vid)?;
                let terms: Option<Vec<(StmtId, i64)>> = view
                    .terms
                    .iter()
                    .map(|(r, k)| stmt_of(*r).map(|s| (s, *k)))
                    .collect();
                Some((key, terms?, view.offset))
            })
            .collect();
        affine.sort_by_key(|(k, _, _)| *k);
        let mut interval: Vec<(StmtId, Interval)> = self
            .interval
            .iter()
            .filter_map(|(vid, iv)| Some((stmt_of(*vid)?, *iv)))
            .collect();
        interval.sort_by_key(|(k, _)| *k);
        let mut layout: Vec<(u32, LayoutKind)> =
            self.layout.iter().map(|(k, v)| (*k, *v)).collect();
        layout.sort_by_key(|(k, _)| *k);
        let mut compaction: Vec<(u32, usize, CompactionWire)> = self
            .compaction
            .iter()
            .filter_map(|((seg, lane), info)| {
                Some((
                    *seg,
                    *lane,
                    CompactionWire {
                        keep: map_all(&info.keep)?,
                        source: map_all(&info.source)?,
                        prefix: map_all(&info.prefix)?,
                        count: stmt_of(info.count)?,
                        sorted: info.sorted,
                    },
                ))
            })
            .collect();
        compaction.sort_by_key(|(seg, lane, _)| (*seg, *lane));
        let mut write_version: Vec<(u32, u64)> =
            self.write_version.iter().map(|(k, v)| (*k, *v)).collect();
        write_version.sort_by_key(|(k, _)| *k);
        let mut segment_hull: Vec<(u32, Option<Interval>)> =
            self.segment_hull.iter().map(|(k, v)| (*k, *v)).collect();
        segment_hull.sort_by_key(|(k, _)| *k);
        let mut runtime_extents: Vec<(u32, Vec<StmtId>)> = self
            .runtime_extents
            .iter()
            .filter_map(|(seg, vids)| Some((*seg, map_all(vids)?)))
            .collect();
        runtime_extents.sort_by_key(|(k, _)| *k);
        let mut clamps: Vec<(StmtId, StmtId, i64, bool)> = self
            .clamps
            .iter()
            .filter_map(|(vid, cl)| {
                Some((stmt_of(*vid)?, stmt_of(cl.source)?, cl.bound, cl.is_min))
            })
            .collect();
        clamps.sort_by_key(|(k, _, _, _)| *k);
        let mut block_views: Vec<(StmtId, BlockView)> = self
            .block_views
            .iter()
            .filter_map(|(vid, bv)| Some((stmt_of(*vid)?, *bv)))
            .collect();
        block_views.sort_by_key(|(k, _)| *k);
        ProvenanceWire {
            affine,
            interval,
            layout,
            compaction,
            write_version,
            segment_hull,
            runtime_extents,
            clamps,
            block_views,
        }
    }

    /// Rebuild from the wire form against `vid_of` (stmt → its value id
    /// in the RECEIVING graph). Entries with unresolvable stmt ids are
    /// dropped.
    pub fn from_wire(w: &ProvenanceWire, vid_of: &dyn Fn(StmtId) -> Option<ValueId>) -> Self {
        let map_all = |stmts: &[StmtId]| -> Option<Vec<ValueId>> {
            stmts.iter().map(|s| vid_of(*s)).collect()
        };
        let mut t = ProvenanceTable::default();
        for (key, terms, offset) in &w.affine {
            let Some(vid) = vid_of(*key) else { continue };
            let Some(terms): Option<Vec<(ValueId, i64)>> =
                terms.iter().map(|(s, k)| vid_of(*s).map(|v| (v, *k))).collect()
            else {
                continue;
            };
            t.affine.insert(
                vid,
                AffineView {
                    terms,
                    offset: *offset,
                },
            );
        }
        for (key, iv) in &w.interval {
            if let Some(vid) = vid_of(*key) {
                t.interval.insert(vid, *iv);
            }
        }
        t.layout = w.layout.iter().copied().collect();
        for (seg, lane, info) in &w.compaction {
            let (Some(keep), Some(source), Some(prefix), Some(count)) = (
                map_all(&info.keep),
                map_all(&info.source),
                map_all(&info.prefix),
                vid_of(info.count),
            ) else {
                continue;
            };
            t.compaction.insert(
                (*seg, *lane),
                CompactionInfo {
                    keep,
                    source,
                    prefix,
                    count,
                    sorted: info.sorted,
                },
            );
        }
        t.write_version = w.write_version.iter().copied().collect();
        t.segment_hull = w.segment_hull.iter().copied().collect();
        for (seg, stmts) in &w.runtime_extents {
            if let Some(vids) = map_all(stmts) {
                t.runtime_extents.insert(*seg, vids);
            }
        }
        for (key, src, bound, is_min) in &w.clamps {
            if let (Some(vid), Some(source)) = (vid_of(*key), vid_of(*src)) {
                t.clamps.insert(
                    vid,
                    ClampView {
                        source,
                        bound: *bound,
                        is_min: *is_min,
                    },
                );
            }
        }
        for (key, bv) in &w.block_views {
            if let Some(vid) = vid_of(*key) {
                t.block_views.insert(vid, *bv);
            }
        }
        t
    }
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
            && self.mask_read_views.is_empty()
            && self.runtime_extents.is_empty()
            && self.clamps.is_empty()
            && self.block_views.is_empty()
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
        self.mask_read_views.remove(&segment);
        self.compaction.retain(|(seg, _), _| *seg != segment);
        self.runtime_extents.remove(&segment);
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

    pub fn runtime_extents_of(&self, segment: u32) -> Option<&Vec<ValueId>> {
        self.runtime_extents.get(&segment)
    }

    pub fn set_runtime_extents(&mut self, segment: u32, extents: Vec<ValueId>) {
        self.runtime_extents.insert(segment, extents);
    }

    // ── Record accessors ─────────────────────────────────────────────

    pub fn slice_view(&self, segment: u32) -> Option<&SliceView> {
        self.slice_views.get(&segment)
    }

    pub fn set_slice_view(&mut self, segment: u32, view: SliceView) {
        self.slice_views.insert(segment, view);
    }

    pub fn mask_read_view(&self, segment: u32) -> Option<&MaskReadView> {
        self.mask_read_views.get(&segment)
    }

    pub fn set_mask_read_view(&mut self, segment: u32, view: MaskReadView) {
        self.mask_read_views.insert(segment, view);
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

    pub fn block_view_of(&self, vid: ValueId) -> Option<BlockView> {
        self.block_views.get(&vid).copied()
    }

    pub fn set_block_view(&mut self, vid: ValueId, view: BlockView) {
        self.block_views.insert(vid, view);
    }

    pub fn clamp_of(&self, vid: ValueId) -> Option<ClampView> {
        self.clamps.get(&vid).copied()
    }

    pub fn set_clamp(&mut self, vid: ValueId, view: ClampView) {
        self.clamps.insert(vid, view);
    }

    pub fn layout_of(&self, segment: u32) -> Option<LayoutKind> {
        self.layout.get(&segment).copied()
    }

    pub fn set_layout(&mut self, segment: u32, kind: LayoutKind) {
        self.layout.insert(segment, kind);
    }

    pub fn compaction_of(&self, segment: u32, lane: usize) -> Option<&CompactionInfo> {
        self.compaction.get(&(segment, lane))
    }

    pub fn set_compaction(&mut self, segment: u32, lane: usize, info: CompactionInfo) {
        self.compaction.insert((segment, lane), info);
    }

    // ── Pass hooks ───────────────────────────────────────────────────

    /// A pass replaced every use of `old` with `new` (duplicate-code
    /// elimination): re-key records and rewrite term lists. Delegates to
    /// [`remap_values`](Self::remap_values) with a singleton map so there is
    /// only one implementation.
    pub fn remap_value(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let mut map = HashMap::with_capacity(1);
        map.insert(old, new);
        self.remap_values(&map);
    }

    /// Batch form of [`remap_value`](Self::remap_value): a pass replaced
    /// every use of each `old` key with its `new` value. Does a SINGLE pass
    /// over the table — re-keying records and rewriting term lists via map
    /// lookups — so cost is O(table + map) instead of O(table) per pair.
    ///
    /// Re-keying preserves first-wins-on-collision (`or_insert`) semantics:
    /// if two `old` keys remap onto the same `new` key, the entry already
    /// present (or remapped first) is kept.
    pub fn remap_values(&mut self, map: &HashMap<ValueId, ValueId>) {
        if map.is_empty() {
            return;
        }
        let lookup = |v: ValueId| map.get(&v).copied().unwrap_or(v);

        // Re-key the value-keyed maps. Collect the records whose key moves,
        // then re-insert with `or_insert` so existing/earlier entries win.
        Self::remap_keyed(&mut self.affine, &lookup);
        Self::remap_keyed(&mut self.interval, &lookup);
        Self::remap_keyed(&mut self.clamps, &lookup);
        Self::remap_keyed(&mut self.block_views, &lookup);

        // Rewrite vids embedded inside record bodies.
        for view in self.clamps.values_mut() {
            view.source = lookup(view.source);
        }
        for exts in self.runtime_extents.values_mut() {
            for vid in exts.iter_mut() {
                *vid = lookup(*vid);
            }
        }
        for view in self.affine.values_mut() {
            for (vid, _) in view.terms.iter_mut() {
                *vid = lookup(*vid);
            }
        }
        for info in self.compaction.values_mut() {
            for vid in info
                .keep
                .iter_mut()
                .chain(info.source.iter_mut())
                .chain(info.prefix.iter_mut())
            {
                *vid = lookup(*vid);
            }
            info.count = lookup(info.count);
        }
    }

    /// Re-key a `ValueId`-keyed map by `lookup`, keeping the first-inserted
    /// entry on collision (mirrors the old `remove`+`or_insert` dance).
    fn remap_keyed<V>(table: &mut HashMap<ValueId, V>, lookup: &dyn Fn(ValueId) -> ValueId) {
        // Pull out only the entries whose key actually moves, leaving the
        // rest (including any that are remap targets) in place so they win.
        let moved: Vec<(ValueId, V)> = table
            .keys()
            .copied()
            .filter(|&k| lookup(k) != k)
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(|k| table.remove(&k).map(|v| (lookup(k), v)))
            .collect();
        for (new_key, v) in moved {
            table.entry(new_key).or_insert(v);
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
                && info.source.iter().all(|v| live(*v))
                && info.prefix.iter().all(|v| live(*v))
        });
        self.slice_views.retain(|_, view| {
            view.start.value_id().map(|v| live(v)).unwrap_or(true)
        });
        self.mask_read_views.retain(|_, view| {
            view.mask_cells
                .iter()
                .chain(view.sources.iter())
                .all(|c| c.value_id().map(|v| live(v)).unwrap_or(true))
        });
        self.runtime_extents
            .retain(|_, exts| exts.iter().all(|v| live(*v)));
        self.clamps
            .retain(|vid, view| live(*vid) && live(view.source));
        self.block_views.retain(|vid, _| live(*vid));
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
        for ((seg, _lane), info) in &self.compaction {
            for v in info
                .keep
                .iter()
                .chain(info.source.iter())
                .chain(info.prefix.iter())
            {
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
        for (seg, exts) in &self.runtime_extents {
            for v in exts {
                if !value_exists(*v) {
                    return Err(format!(
                        "runtime-extents record for segment {} references dead value {:?}",
                        seg, v
                    ));
                }
            }
        }
        for vid in self.block_views.keys() {
            if !value_exists(*vid) {
                return Err(format!(
                    "block-view record references dead value {:?}",
                    vid
                ));
            }
        }
        for (vid, view) in &self.clamps {
            if !value_exists(*vid) || !value_exists(view.source) {
                return Err(format!(
                    "clamp record {:?} -> {:?} references a dead value",
                    vid, view
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
        for (seg, view) in &self.mask_read_views {
            for cell in view.mask_cells.iter().chain(view.sources.iter()) {
                if let Some(vid) = cell.value_id() {
                    if !value_exists(vid) {
                        return Err(format!(
                            "mask-read view for segment {} references dead value {:?}",
                            seg, vid
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
