//! The interface catalogue of `revision/phase1/rules/README.md` §4.
//!
//! An *interface* says how a group of wires represents a value, capacity
//! included: two interfaces that differ in any parameter are two
//! requirements (§3.1). This module is the whole catalogue as Rust types —
//! the families of §4.1 to §4.5, the layout λ of §4.1 and the μ grammar of
//! §4.2 — defined now even where step 1 has no producer for them, so that a
//! later step adds a rule and not a type.
//!
//! Every capacity, width and count is an integer ([`Capacity`]): phase 1
//! leaves no symbolic variable on the graph (`../README.md` §2,
//! "Invariants").

use std::fmt::Write as _;

use crate::phase0::desc::{Dtype, ValueKey};
use crate::phase0::term::{TermArena, TermId};

/// A capacity, query count or bit width. Always a concrete integer by the
/// time it reaches an interface.
pub type Capacity = i128;

/// The interface families of `rules/README.md` §4, which is what a consumer
/// names when it asks for a row: the capacity inside the interface is the
/// provider's to decide (`../README.md` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// §4.1, payload-bearing.
    Store,
    Masked,
    Packed,
    Segments,
    /// §4.2, a view with no payload.
    Ref,
    /// §4.3, one wire.
    Wire,
    IndexNorm,
    /// §4.4, reads.
    Coords,
    ReadReq,
    Values,
    RomTable,
    RomContext,
    /// §4.5, counting, fibers, sources and delivery.
    Ranks,
    Groups,
    Payload,
    Export,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Store => "Store",
            Family::Masked => "Masked",
            Family::Packed => "Packed",
            Family::Segments => "Segments",
            Family::Ref => "Ref",
            Family::Wire => "Wire",
            Family::IndexNorm => "IndexNorm",
            Family::Coords => "Coords",
            Family::ReadReq => "ReadReq",
            Family::Values => "Values",
            Family::RomTable => "RomTable",
            Family::RomContext => "RomContext",
            Family::Ranks => "Ranks",
            Family::Groups => "Groups",
            Family::Payload => "Payload",
            Family::Export => "Export",
        }
    }
}

/// One position-changing operation on the way from a bundle's dense root to
/// the value it holds (`rules/README.md` §4.1, λ.lineage).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LineageStep {
    Filter { mask: ValueKey },
    Slice { lo: TermId, hi: TermId, step: i128, axis: usize },
    Take { index: ValueKey, axis: usize },
    Transpose { permutation: Vec<usize> },
    Reshape { shape: Vec<Capacity> },
}

/// λ.layout of `rules/README.md` §4.1: where a slot's logical coordinate
/// lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayoutKind {
    /// Slot `p` holds the logical coordinate `p` of the capacity grid.
    Box(Vec<Capacity>),
    /// Rank 1: slot `j` holds logical index `j`, valid while `j < ℓ`.
    Compact(Capacity),
    /// Slot `i` holds the element whose logical index is the prefix rank of
    /// the validity wires at `i`; `ranks` names the `Ranks` node when the
    /// validity came from a mask.
    Ranked { capacity: Capacity, ranks: Option<ValueKey> },
    /// Slot `i` holds the logical key on a wire; no order promise.
    Keyed { capacity: Capacity },
}

impl LayoutKind {
    /// Slots the layout holds.
    pub fn slots(&self) -> Capacity {
        match self {
            LayoutKind::Box(grid) => grid.iter().product(),
            LayoutKind::Compact(b)
            | LayoutKind::Ranked { capacity: b, .. }
            | LayoutKind::Keyed { capacity: b } => *b,
        }
    }

    /// *dense* of §4.1: the key of a slot is the slot, a constant.
    pub fn is_dense(&self) -> bool {
        matches!(self, LayoutKind::Box(_) | LayoutKind::Compact(_))
    }

    /// *ordered* of §4.1: the valid slots in physical order are the elements
    /// in logical order.
    pub fn is_ordered(&self) -> bool {
        !matches!(self, LayoutKind::Keyed { .. })
    }

    pub fn tag(&self) -> &'static str {
        match self {
            LayoutKind::Box(_) => "box",
            LayoutKind::Compact(_) => "compact",
            LayoutKind::Ranked { .. } => "ranked",
            LayoutKind::Keyed { .. } => "keyed",
        }
    }
}

/// λ = (layout, root, lineage) (`rules/README.md` §4.1). A dense bundle has
/// an empty lineage and is its own root.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    pub layout: LayoutKind,
    pub root: ValueKey,
    pub lineage: Vec<LineageStep>,
}

impl Layout {
    /// The λ of a bundle that is its own dense root.
    pub fn dense(layout: LayoutKind, root: ValueKey) -> Layout {
        Layout { layout, root, lineage: Vec::new() }
    }
}

/// The μ grammar of `rules/README.md` §4.2: a coordinate expression of a
/// `Ref`'s map. `π_m` is a parameter wire, named by its index in the `Ref`'s
/// parameter list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Mu {
    /// `n`, a literal.
    Const(Capacity),
    /// `i_k`, the k-th input coordinate.
    Coord(usize),
    /// `π_m`, a parameter wire.
    Param(usize),
    Add(Box<Mu>, Box<Mu>),
    Sub(Box<Mu>, Box<Mu>),
    /// `n × e`.
    Scale(Capacity, Box<Mu>),
    /// `π_m × e`.
    ScaleParam(usize, Box<Mu>),
    /// `e div n`, `n ≥ 1`.
    DivConst(Box<Mu>, Capacity),
    /// `e mod n`.
    ModConst(Box<Mu>, Capacity),
    /// `e div π_m`, evaluated by `divmod_d`.
    DivParam(Box<Mu>, usize),
    /// `e mod π_m`.
    ModParam(Box<Mu>, usize),
}

impl Mu {
    /// Whether the map is a compile-time constant: no parameter wire occurs.
    pub fn is_constant(&self) -> bool {
        match self {
            Mu::Const(_) | Mu::Coord(_) => true,
            Mu::Param(_) | Mu::ScaleParam(..) | Mu::DivParam(..) | Mu::ModParam(..) => false,
            Mu::Add(a, b) | Mu::Sub(a, b) => a.is_constant() && b.is_constant(),
            Mu::Scale(_, e) | Mu::DivConst(e, _) | Mu::ModConst(e, _) => e.is_constant(),
        }
    }
}

/// An extent of a `Ref` (`rules/README.md` §4.2, `ext`): a literal, or the
/// term whose wire the `Ref` node raises.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Extent {
    Literal(Capacity),
    Term(TermId),
}

/// Which form μ is in (`mapping/README.md` §2.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MuForm {
    /// `(μ_c, S)`: one expression per owner axis, with the key strides.
    Coordinate { strides: Vec<Capacity> },
    /// One expression giving the owner's key directly; admitted over an
    /// owner whose key is the C-order index, which is every layout but box.
    Key,
}

/// `Ref[v; o, μ]` of §4.2.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RefIface {
    /// The node whose representation the map points into.
    pub owner: OwnerRef,
    pub form: MuForm,
    /// One μ per owner axis (coordinate form), or the one key expression.
    pub map: Vec<Mu>,
    /// The terms the parameter wires `π` hold, in `Mu::Param` order.
    pub params: Vec<TermId>,
    /// `dims_v[k]` per axis of the mapped value.
    pub ext: Vec<Extent>,
    /// How many view operations are folded; bounded by `Ω.map_depth`.
    pub span: u32,
    /// The `Ref` this one is stacked on, when it is *nested*
    /// (`mapping/README.md` §2.3): its map runs over the owner value's
    /// coordinates and this one's over the view's, and `REINDEX_REF` alone
    /// consumes the pair. A flat `Ref` has `None`, and only a flat one is
    /// read (`mapping/README.md` §2.6).
    pub nested: Option<Box<RefIface>>,
}

impl RefIface {
    /// Whether the map mentions no parameter wire, so that composing it with
    /// a constant coordinate folds to a constant (`make_ref.md`).
    pub fn is_constant(&self) -> bool {
        self.map.iter().all(Mu::is_constant)
    }

    /// A `Ref` whose owner is a node of the Store family, which is the only
    /// form a reader evaluates.
    pub fn is_flat(&self) -> bool {
        self.nested.is_none()
    }
}

/// `Store[o, v, τ, d, B; λ]` of §4.1.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StoreIface {
    pub owner: ValueKey,
    pub dtype: Dtype,
    /// `B`, the slot count.
    pub capacity: Capacity,
    pub lambda: Layout,
}

/// The interface of a delivered value (`rules/README.md` §4.5, `Export`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExportIface {
    Scalar(Dtype),
    /// A `Packed` with its ℓ, or a dense box `Store` with its shape wires.
    Array { dtype: Dtype, capacity: Capacity },
}

/// One coordinate of a batch entry: a literal, or a term.
///
/// `rules/README.md` §3.3 writes an entry's coordinate as "a term or a
/// constant". A grid index and the width of a range axis are integers the
/// construction computes, so they are literals here; a slice start and an
/// index element are terms. [`Batch::coordinate_terms`] turns either into
/// the term the entry's coordinate takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Coordinate {
    Literal(Capacity),
    Term(TermId),
}

impl Coordinate {
    pub fn literal(&self) -> Option<Capacity> {
        match self {
            Coordinate::Literal(n) => Some(*n),
            Coordinate::Term(_) => None,
        }
    }

    pub fn term(&self) -> Option<TermId> {
        match self {
            Coordinate::Term(t) => Some(*t),
            Coordinate::Literal(_) => None,
        }
    }
}

/// What decides whether a batch entry counts in a run (`rules/README.md`
/// §3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Live {
    /// The default live term: the coordinate is in range of v on every axis,
    /// `⋀_a (0 ≤ i_a < dims_v[a])`. It is a marker rather than a term id,
    /// because the term depends on v's dims, which the batch does not name;
    /// `ENUM_COORDS` builds the per-axis comparison it needs from the value
    /// it matched.
    InRange,
    /// A Boolean term the consumer supplied.
    Term(crate::phase0::term::BoolId),
}

/// A batch entry: a coordinate tuple and what says whether the entry counts
/// in this run (`rules/README.md` §3.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchEntry {
    pub coordinate: Vec<Coordinate>,
    pub live: Live,
}

/// The batch of a `Fetch` (`rules/README.md` §3.3). A *range batch* writes
/// its entries as a per-axis affine template; a general batch lists them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Batch {
    /// `[()]`, the one entry of a scalar (§3.2).
    Scalar,
    General(Vec<BatchEntry>),
    Range {
        /// Per axis: `i_a = t_a + s_a·j_a` over `[0, W_a)`, optionally
        /// guarded by a broadcast Boolean `b_a`.
        axes: Vec<RangeAxis>,
        live: Live,
    },
}

impl Batch {
    /// `Q`, the number of entries.
    pub fn len(&self) -> Capacity {
        match self {
            Batch::Scalar => 1,
            Batch::General(entries) => entries.len() as Capacity,
            Batch::Range { axes, .. } => axes.iter().map(|a| a.width).product(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The rank of a coordinate of this batch.
    pub fn rank(&self) -> usize {
        match self {
            Batch::Scalar => 0,
            Batch::General(entries) => entries.first().map(|e| e.coordinate.len()).unwrap_or(0),
            Batch::Range { axes, .. } => axes.len(),
        }
    }

    /// The grid index of entry `at` per axis, entries in C-order
    /// (`rules/README.md` §3.3).
    pub fn slot(&self, at: Capacity) -> Vec<Capacity> {
        let Batch::Range { axes, .. } = self else {
            return Vec::new();
        };
        let mut strides: Vec<Capacity> = vec![1; axes.len()];
        for k in (0..axes.len().saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * axes[k + 1].width;
        }
        let mut rest = at;
        let mut out = Vec::with_capacity(axes.len());
        for stride in strides {
            let index = if stride > 0 { rest / stride } else { 0 };
            rest -= index * stride;
            out.push(index);
        }
        out
    }

    /// The coordinate of entry `j` as terms, built through the arena so that
    /// a template with a term base keeps its offset (§3.3).
    pub fn coordinate_terms(&self, terms: &mut TermArena, at: Capacity) -> Vec<TermId> {
        match self {
            Batch::Scalar => Vec::new(),
            Batch::General(entries) => entries
                .get(at as usize)
                .map(|entry| {
                    entry
                        .coordinate
                        .iter()
                        .map(|coordinate| match coordinate {
                            Coordinate::Literal(value) => terms.constant(*value),
                            Coordinate::Term(t) => *t,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Batch::Range { axes, .. } => {
                let slot = self.slot(at);
                axes.iter()
                    .zip(slot)
                    .map(|(axis, index)| axis.term_at(terms, index))
                    .collect()
            }
        }
    }

    /// The leaf term of each axis that every entry shares: the template's
    /// base for a range batch, the entry's own coordinate otherwise. CONST
    /// is asked on these, which is what makes a range batch one question per
    /// axis (§3.3).
    pub fn axis_bases(&self, terms: &mut TermArena, at: Capacity) -> Vec<TermId> {
        match self {
            Batch::Range { axes, .. } => {
                axes.iter().map(|axis| axis.base_term(terms)).collect()
            }
            _ => self.coordinate_terms(terms, at),
        }
    }

    /// The liveness of the batch, which every entry shares for a range batch.
    pub fn live(&self) -> Live {
        match self {
            Batch::Scalar => Live::InRange,
            Batch::General(entries) => {
                entries.first().map(|entry| entry.live).unwrap_or(Live::InRange)
            }
            Batch::Range { live, .. } => *live,
        }
    }

    /// The grid batch of a capacity box (`representation/README.md` §2):
    /// `i_{p,a} = p_a` over `∏_a [0, B_a)` with the default live template.
    pub fn grid(capacities: &[Capacity]) -> Batch {
        Batch::Range {
            axes: capacities
                .iter()
                .map(|&width| RangeAxis {
                    base: Coordinate::Literal(0),
                    stride: 1,
                    width,
                    broadcast: None,
                })
                .collect(),
            live: Live::InRange,
        }
    }
}

/// One axis of a range batch (`rules/README.md` §3.3).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RangeAxis {
    pub base: Coordinate,
    pub stride: Capacity,
    pub width: Capacity,
    /// `ite(b, 0, t + s·j)` for a broadcast axis.
    pub broadcast: Option<crate::phase0::term::BoolId>,
}

impl RangeAxis {
    /// The coordinate at grid index `j`, structurally: `t + s·j` folded when
    /// `t` is a literal, and the bare term when the offset is zero. A term
    /// base with a non-zero offset has no structural form, so the caller
    /// that needs the value takes [`RangeAxis::term_at`] instead.
    pub fn at(&self, j: Capacity) -> Option<Coordinate> {
        match self.base {
            Coordinate::Literal(base) => Some(Coordinate::Literal(base + self.stride * j)),
            Coordinate::Term(t) if self.stride * j == 0 => Some(Coordinate::Term(t)),
            Coordinate::Term(_) => None,
        }
    }

    /// The coordinate at grid index `j` as a term: `t + s·j`, built through
    /// the arena, which folds it to a literal when the base is one.
    ///
    /// A broadcast axis reads position 0 whenever `b` holds, so its
    /// coordinate is `ite(b, 0, t + s·j)`.
    pub fn term_at(&self, terms: &mut TermArena, j: Capacity) -> TermId {
        let base = match self.base {
            Coordinate::Literal(value) => terms.constant(value),
            Coordinate::Term(t) => t,
        };
        let offset = terms.constant(self.stride * j);
        let position = terms.add(base, offset);
        match self.broadcast {
            None => position,
            Some(b) => {
                let zero = terms.constant(0);
                terms.ite(b, zero, position)
            }
        }
    }

    /// The base of the template as a term, which is the coordinate at grid
    /// index 0 and the one leaf every entry of the axis shares.
    pub fn base_term(&self, terms: &mut TermArena) -> TermId {
        self.term_at(terms, 0)
    }

    /// Whether the axis is the identity template `j ↦ j`.
    pub fn is_identity(&self) -> bool {
        self.stride == 1 && matches!(self.base, Coordinate::Literal(0))
    }
}

/// The node that holds the elements a read addresses: phase 1's name for the
/// *owner node* of `rules/README.md` §4.2 and §4.4.
///
/// A context's identity is "(o, B)", the owner node and its capacity. A node
/// is named here by the value it represents, the family, the capacity and
/// the layout of its bundle: two providers of one value that differ in any
/// of these are two owners, with two read requests and two ROM contexts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OwnerRef {
    pub value: ValueKey,
    pub family: Family,
    pub capacity: Capacity,
    pub layout: LayoutKind,
    /// `canon` of a `Packed` (§4.1): the canonical and the plain form of one
    /// payload are two nodes, so a context opened on one is not the other's.
    pub canon: bool,
}

impl OwnerRef {
    /// `B_tot`, the slots of the owner (`reading/README.md` §3.2).
    pub fn slots(&self) -> Capacity {
        self.capacity
    }

    /// *dense* of §4.1: the slot of a key is the key, a constant.
    pub fn is_dense(&self) -> bool {
        self.layout.is_dense()
    }

    pub fn render(&self) -> String {
        format!(
            "{}:{}[{}]{}",
            self.value.encode(),
            self.layout.tag(),
            self.capacity,
            if self.canon { " canon" } else { "" }
        )
    }
}

/// What a `Ranks` row is about (`rules/README.md` §4.5): the bundle I whose
/// positions are ranked, and which bits are ranked.
///
/// `Ranks(I)` ranks the mask's value wires, `Ranks(I, valid)` the bundle's
/// own validity wires, and the two are different requirements over one
/// bundle. Together with the value the row names — the *mask identity* of
/// `masks/README.md` §2, which the provider table reaches through the alias
/// chain — this is the identity `(mask identity, I)` §4.5 gives the node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RanksOf {
    /// The bundle whose positions are ranked.
    pub owner: OwnerRef,
    /// `Ranks(I, valid)`: rank I's validity wires rather than its values.
    pub valid: bool,
}

impl RanksOf {
    pub fn render(&self) -> String {
        format!(
            "{}{}",
            self.owner.render(),
            if self.valid { ", valid" } else { "" }
        )
    }
}

/// The compile-time flags of a `Groups` bundle (`rules/README.md` §4.5).
///
/// They are parameters of the interface, so two layouts of one value's
/// fibers that differ in a flag are two requirements (§3.1) and the fold
/// above each knows which gadget it may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupFlags {
    /// Increasing `j` is the source order within a fiber.
    pub ordered: bool,
    /// Every fiber's valid slots come first.
    pub prefix: bool,
    /// Every slot with `g = 0` holds `enc(0)`.
    pub zeroed: bool,
}

/// `Groups[v, A; P, W]` of §4.5.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupsIface {
    pub owner: ValueKey,
    /// `A`, the axes the reduction runs over; part of the row key.
    pub axes: Vec<usize>,
    /// `P`, the fibers.
    pub fibers: Capacity,
    /// `W`, the slots per fiber.
    pub width: Capacity,
    pub dtype: Dtype,
    pub flags: GroupFlags,
    /// `slots`, the λ of the P·W slot grid: what `CUMSUM` produces in.
    pub slots: Layout,
}

/// A concrete interface: a family with every parameter filled in.
///
/// Equality and hashing are structural, which is what makes a requirement
/// node unique per interface (`rules/README.md` §3.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Interface {
    /// §4.1.
    Store(StoreIface),
    Masked { dtype: Dtype, capacity: Capacity, lambda: Layout },
    Packed { dtype: Dtype, capacity: Capacity, lambda: Layout, canon: bool },
    Segments { owner: ValueKey, dtype: Dtype, capacity: Capacity, segments: usize },
    /// §4.2.
    Ref(RefIface),
    /// §4.3.
    Wire(Dtype),
    IndexNorm { extent: TermId },
    /// §4.4.
    Coords { queries: Capacity, rank: usize },
    ReadReq {
        owner: OwnerRef,
        queries: Capacity,
        /// Whether every address folded to a compile-time constant, which is
        /// `read_direct.md`'s D1 "from the record, no question". It is a
        /// function of the value, the batch and the provider, all of which
        /// this interface names, so it makes no second requirement.
        constant: bool,
    },
    Values { dtype: Dtype, queries: Capacity },
    RomTable { owner: OwnerRef },
    RomContext { owner: OwnerRef },
    /// §4.5.
    Ranks { subject: RanksOf, capacity: Capacity, width: Capacity },
    Groups(GroupsIface),
    Payload { dtype: Dtype, capacity: Capacity, grid: Vec<Capacity> },
    Export(ExportIface),
}

impl Interface {
    pub fn family(&self) -> Family {
        match self {
            Interface::Store(_) => Family::Store,
            Interface::Masked { .. } => Family::Masked,
            Interface::Packed { .. } => Family::Packed,
            Interface::Segments { .. } => Family::Segments,
            Interface::Ref(_) => Family::Ref,
            Interface::Wire(_) => Family::Wire,
            Interface::IndexNorm { .. } => Family::IndexNorm,
            Interface::Coords { .. } => Family::Coords,
            Interface::ReadReq { .. } => Family::ReadReq,
            Interface::Values { .. } => Family::Values,
            Interface::RomTable { .. } => Family::RomTable,
            Interface::RomContext { .. } => Family::RomContext,
            Interface::Ranks { .. } => Family::Ranks,
            Interface::Groups(_) => Family::Groups,
            Interface::Payload { .. } => Family::Payload,
            Interface::Export(_) => Family::Export,
        }
    }

    /// The λ of a Store-family bundle, for the alignment test of §4.1.
    pub fn lambda(&self) -> Option<&Layout> {
        match self {
            Interface::Store(s) => Some(&s.lambda),
            Interface::Masked { lambda, .. } | Interface::Packed { lambda, .. } => Some(lambda),
            _ => None,
        }
    }

    /// A short rendering for reports and the JSON export.
    pub fn render(&self) -> String {
        let mut out = String::new();
        match self {
            Interface::Store(s) => {
                let _ = write!(out, "Store[{}, {}; {}]", s.dtype.as_str(), s.capacity,
                               s.lambda.layout.tag());
            }
            Interface::Masked { dtype, capacity, .. } => {
                let _ = write!(out, "Masked[{}, {}]", dtype.as_str(), capacity);
            }
            Interface::Packed { dtype, capacity, canon, .. } => {
                let _ = write!(out, "Packed[{}, {}, canon={}]", dtype.as_str(), capacity, canon);
            }
            Interface::Segments { dtype, capacity, segments, .. } => {
                let _ = write!(out, "Segments[{}, {}, n={}]", dtype.as_str(), capacity, segments);
            }
            Interface::Ref(r) => {
                let _ = write!(
                    out,
                    "Ref[owner {}, span {}{}]",
                    r.owner.render(),
                    r.span,
                    if r.is_flat() { "" } else { ", nested" }
                );
            }
            Interface::Wire(d) => {
                let _ = write!(out, "Wire[{}]", d.as_str());
            }
            Interface::IndexNorm { .. } => out.push_str("IndexNorm"),
            Interface::Coords { queries, rank } => {
                let _ = write!(out, "Coords[{}, rank {}]", queries, rank);
            }
            Interface::ReadReq { owner, queries, constant } => {
                let _ = write!(
                    out,
                    "ReadReq[{}, {}{}]",
                    owner.render(),
                    queries,
                    if *constant { ", const" } else { "" }
                );
            }
            Interface::Values { dtype, queries } => {
                let _ = write!(out, "Values[{}, {}]", dtype.as_str(), queries);
            }
            Interface::RomTable { owner } => {
                let _ = write!(out, "RomTable[{}]", owner.render());
            }
            Interface::RomContext { owner } => {
                let _ = write!(out, "RomContext[{}]", owner.render());
            }
            Interface::Ranks { subject, capacity, width } => {
                let _ = write!(
                    out,
                    "Ranks[{}; {}, w {}]",
                    subject.render(),
                    capacity,
                    width
                );
            }
            Interface::Groups(groups) => {
                let _ = write!(
                    out,
                    "Groups[{}x{}, A {:?}]",
                    groups.fibers, groups.width, groups.axes
                );
            }
            Interface::Payload { dtype, capacity, .. } => {
                let _ = write!(out, "Payload[{}, {}]", dtype.as_str(), capacity);
            }
            Interface::Export(ExportIface::Scalar(d)) => {
                let _ = write!(out, "Export[Wire {}]", d.as_str());
            }
            Interface::Export(ExportIface::Array { dtype, capacity }) => {
                let _ = write!(out, "Export[{} x {}]", dtype.as_str(), capacity);
            }
        }
        out
    }
}

/// Alignment of two Store-family bundles (`rules/README.md` §4.1): decided
/// by construction, never by a solver beyond the per-axis `PROVE` the caller
/// supplies through `dims_equal`.
pub fn aligned(a: &Layout, b: &Layout, mut dims_equal: impl FnMut() -> bool) -> bool {
    match (&a.layout, &b.layout) {
        (LayoutKind::Box(x), LayoutKind::Box(y)) => x == y && dims_equal(),
        (LayoutKind::Compact(x), LayoutKind::Compact(y)) => x == y && dims_equal(),
        (LayoutKind::Ranked { capacity: x, .. }, LayoutKind::Ranked { capacity: y, .. })
        | (LayoutKind::Keyed { capacity: x }, LayoutKind::Keyed { capacity: y }) => {
            x == y && a.lineage == b.lineage && (a.root == b.root || dims_equal())
        }
        _ => false,
    }
}
