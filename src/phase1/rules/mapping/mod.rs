//! Rule group `mapping` (`revision/phase1/rules/mapping/README.md`).
//!
//! The group answers `Represent(v, Ref[v; o, μ])`: it makes a value readable
//! as a view of another value's representation, with no payload of its own.
//! This module holds what the three rule files share — the normalised source
//! view of §3, the key strides of §2.2, and the map μ of §2.3 — and nothing
//! a rule file does not state.
//!
//! **Where a slice start comes from.** A slice start is the term
//! `L = norm(t_lo, dims_u[a])` phase 0 built inside the slice length
//! (`../../phase0/rules/helpers.rs`). The group reads it back out of that
//! length term ([`slice_lower`]) rather than building a second copy: phase 0
//! already interned it, so the read-back gives the very term id every other
//! consumer of the slice shares, and it needs no endpoint from the AST. A
//! slice whose length term has another shape is no node of this group, and
//! the row says so.

pub(crate) mod make_ref;
pub(crate) mod make_ref_runtime;
pub(crate) mod reindex_ref;

use crate::phase0::desc::ValueKey;
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::{NodeView, TermId};

use crate::phase1::ast_index::{AstNode, SubscriptEntry};
use crate::phase1::build::Builder;
use crate::phase1::interface::{Capacity, Extent, Family, Interface, Mu};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::{Condition, Question};
use crate::phase1::rules::{parameter, Candidate};
use crate::phase1::table::RowRequest;

/// The families a `Ref`'s owner may be held in (`make_ref.md`,
/// `required_input_ports`).
pub const OWNERS: [Family; 4] =
    [Family::Store, Family::Masked, Family::Packed, Family::Ref];

/// One entry of a normalised subscript (`make_ref.md`, `source_pattern`):
/// the entries that consume an owner axis, in owner-axis order, with the
/// `None` entries that insert one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// `:`, the whole axis.
    Whole,
    /// `lo:hi:step`, with the literal step.
    Slice { step: i64 },
    /// An integer scalar index.
    Index,
    /// `None`: an output axis of extent 1 over no owner axis.
    NewAxis,
}

/// The source view a value of Δ is (`mapping/README.md` §3): which family of
/// `make_ref.md`'s `source_pattern` matched, with its constant parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    /// A subscript with basic entries, normalised to one entry per owner
    /// axis plus the `None` entries.
    Subscript(Vec<Entry>),
    /// `transpose(u, p)`, `.T`, `swapaxes`, `moveaxis`: the permutation `p`
    /// with `μ_c,p_j = i_j`.
    Transpose(Vec<usize>),
    /// `reshape`, `flatten`, `ravel`: C-order preserves the linear position.
    Reshape,
    /// `broadcast_to`: owner axis `a` aligns with output axis `a + r_v − r_u`.
    Broadcast,
    /// `diag(u)` on a rank-2 `u`: the diagonal, `μ_c = (i_0, i_0)`.
    Diagonal,
}

/// What the source pattern of the two `MAKE_REF` rules read: the value, its
/// owner, and the view.
#[derive(Debug, Clone)]
pub struct Match {
    pub value: ValueKey,
    pub owner: ValueKey,
    pub view: View,
    /// `dims_v`, one term per axis of the view.
    pub dims: Vec<TermId>,
    /// `dims_u`, one term per axis of the owner value.
    pub owner_dims: Vec<TermId>,
}

/// The transfer rules the two `MAKE_REF` rules cover (`make_ref.md`).
const COVERED: [Phase0Rule; 10] = [
    Phase0Rule::ESlice,
    Phase0Rule::EIndex,
    Phase0Rule::ENewAxis,
    Phase0Rule::ESqueeze,
    Phase0Rule::ETranspose,
    Phase0Rule::EMoveaxis,
    Phase0Rule::EReshape,
    Phase0Rule::EBroadcast,
    Phase0Rule::EReverse,
    Phase0Rule::EDiag,
];

/// `source_pattern` of `make_ref.md` and `make_ref_runtime.md`, which is one
/// pattern: the AST node and the producer chain of v.
///
/// Returns `None` when v is no view of a single array value, which includes
/// every subscript with an advanced entry (`GATHER`'s) and every
/// mask (`BUILD_MASKED`'s).
pub fn matches(builder: &Builder<'_>, request: &RowRequest) -> Option<Match> {
    let key = request.target.as_value()?.clone();
    let entry = builder.entry(&key)?;
    let Producer::Applied { rules, operands, .. } = &entry.producer else {
        return None;
    };
    if !rules.iter().any(|rule| COVERED.contains(rule)) {
        return None;
    }
    // A subscript that also filtered, took or reduced is another group's.
    if rules.iter().any(|rule| !COVERED.contains(rule)) {
        return None;
    }
    let array = entry.desc.as_array()?;
    if array.rank() == 0 || !super::sources::has_encoding(array.dtype) {
        return None;
    }
    let owner = operands
        .iter()
        .find(|operand| builder.array(operand).is_some())?
        .clone();
    let owner_dims = builder.array(&owner)?.dims.clone();
    let view = view_of(builder, &key, rules, array.rank(), owner_dims.len())?;
    Some(Match {
        value: key,
        owner,
        view,
        dims: array.dims.clone(),
        owner_dims,
    })
}

/// Which family of `make_ref.md`'s `source_pattern` the node is, with the
/// constant parameters read off the AST node.
fn view_of(
    builder: &Builder<'_>,
    key: &ValueKey,
    rules: &[Phase0Rule],
    rank: usize,
    owner_rank: usize,
) -> Option<View> {
    // `diag` needs no parameter and no AST node: the direction is decided
    // by the two ranks. Only the rank-2 form is a view; `diag` of a rank-1
    // array writes a matrix whose off-diagonal slots are 0 and is no map
    // (`make_ref.md`, Open).
    if rules.contains(&Phase0Rule::EDiag) {
        return (owner_rank == 2 && rank == 1).then_some(View::Diagonal);
    }
    let node = builder.node(key)?;
    if rules.contains(&Phase0Rule::EReshape) {
        return Some(View::Reshape);
    }
    if rules.contains(&Phase0Rule::EBroadcast) {
        return Some(View::Broadcast);
    }
    if rules.contains(&Phase0Rule::ETranspose) || rules.contains(&Phase0Rule::EMoveaxis) {
        return Some(View::Transpose(permutation(node, owner_rank)?));
    }
    Some(View::Subscript(normalise(node, owner_rank, rank)?))
}

/// The permutation of a transpose, in `μ_c,p_j = i_j` form (`make_ref.md`).
///
/// `transpose(u, p)` and `.T` are read from the call; `swapaxes(x, y)` is the
/// identity with two axes exchanged and `moveaxis(src, dst)` the archive's
/// B.1 construction.
fn permutation(node: &AstNode, owner_rank: usize) -> Option<Vec<usize>> {
    let axis = |value: i64| -> Option<usize> {
        let wrapped = if value < 0 { value + owner_rank as i64 } else { value };
        (0..owner_rank as i64).contains(&wrapped).then_some(wrapped as usize)
    };
    match node.member()? {
        "transpose" | "T" => match parameter(node, 1, "axes").and_then(|value| value.ints()) {
            Some(axes) if axes.len() == owner_rank => {
                axes.into_iter().map(axis).collect::<Option<Vec<_>>>()
            }
            // The default is the reversal of the axis order.
            _ => Some((0..owner_rank).rev().collect()),
        },
        "swapaxes" => {
            let (x, y) = (
                axis(parameter(node, 1, "axis1")?.int()?)?,
                axis(parameter(node, 2, "axis2")?.int()?)?,
            );
            let mut order: Vec<usize> = (0..owner_rank).collect();
            order.swap(x, y);
            Some(order)
        }
        "moveaxis" => {
            let source = axis(parameter(node, 1, "source")?.int()?)?;
            let destination = axis(parameter(node, 2, "destination")?.int()?)?;
            let mut rest: Vec<usize> = (0..owner_rank).filter(|&a| a != source).collect();
            rest.insert(destination.min(rest.len()), source);
            // `rest[j]` is the owner axis at output axis j, and μ wants the
            // output axis at each owner axis.
            let mut order = vec![0; owner_rank];
            for (output, &owner) in rest.iter().enumerate() {
                order[owner] = output;
            }
            Some(order)
        }
        _ => None,
    }
}

/// The normalised entries of a subscript (`make_ref.md`, `source_pattern`):
/// the Ellipsis expanded to `:` entries, `:` appended until the entries that
/// consume an owner axis number `r_u`, and the method forms written as the
/// entry they are.
fn normalise(node: &AstNode, owner_rank: usize, rank: usize) -> Option<Vec<Entry>> {
    let mut out: Vec<Entry> = Vec::new();
    match node {
        AstNode::Subscript { entries } => {
            let consuming = entries
                .iter()
                .filter(|entry| {
                    !matches!(entry, SubscriptEntry::NewAxis | SubscriptEntry::Ellipsis)
                })
                .count();
            let missing = owner_rank.saturating_sub(consuming);
            for entry in entries {
                match entry {
                    SubscriptEntry::Single { .. } => out.push(Entry::Index),
                    SubscriptEntry::Slice { step, .. } => {
                        out.push(Entry::Slice { step: step.unwrap_or(1) })
                    }
                    SubscriptEntry::NewAxis => out.push(Entry::NewAxis),
                    SubscriptEntry::Ellipsis => {
                        out.extend(std::iter::repeat(Entry::Whole).take(missing))
                    }
                }
            }
            if !entries.iter().any(|entry| matches!(entry, SubscriptEntry::Ellipsis)) {
                out.extend(std::iter::repeat(Entry::Whole).take(missing));
            }
        }
        // `squeeze(u, k)` is the index 0 at k; `expand_dims(u, k)` a `None`
        // at k; `flip` the entry `::-1`; `atleast_*d` leading `None` entries.
        AstNode::Call { member, .. } | AstNode::MethodCall { member, .. } => {
            let axes = |default: Vec<i64>| -> Vec<usize> {
                parameter(node, 1, "axis")
                    .and_then(|value| value.ints())
                    .unwrap_or(default)
                    .into_iter()
                    .map(|a| if a < 0 { a + owner_rank as i64 } else { a })
                    .filter(|a| (0..owner_rank as i64).contains(a))
                    .map(|a| a as usize)
                    .collect()
            };
            match member.as_str() {
                "squeeze" => {
                    let dropped = axes((0..owner_rank as i64).collect());
                    for axis in 0..owner_rank {
                        out.push(if dropped.contains(&axis) {
                            Entry::Index
                        } else {
                            Entry::Whole
                        });
                    }
                }
                "expand_dims" => {
                    let inserted = axes(vec![0]);
                    for axis in 0..=owner_rank {
                        if inserted.contains(&axis) {
                            out.push(Entry::NewAxis);
                        }
                        if axis < owner_rank {
                            out.push(Entry::Whole);
                        }
                    }
                }
                "atleast_1d" | "atleast_2d" | "atleast_3d" => {
                    for _ in owner_rank..rank {
                        out.push(Entry::NewAxis);
                    }
                    out.extend(std::iter::repeat(Entry::Whole).take(owner_rank));
                }
                "flip" | "flipud" | "fliplr" => {
                    let reversed = match member.as_str() {
                        "flipud" => vec![0],
                        "fliplr" => vec![1],
                        _ => axes((0..owner_rank as i64).collect()),
                    };
                    for axis in 0..owner_rank {
                        out.push(if reversed.contains(&axis) {
                            Entry::Slice { step: -1 }
                        } else {
                            Entry::Whole
                        });
                    }
                }
                _ => return None,
            }
        }
        _ => return None,
    }
    Some(out)
}

/// The key strides `S^o_a` of `mapping/README.md` §2.2, when every one of
/// them is a literal.
///
/// A box owner has the capacity strides, which are literal by construction; a
/// compact owner or a flat `Ref` has `∏_{b>a} dims_u[b]`, literal when every
/// inner extent is. A ranked or keyed owner is rank 1 with the stride 1.
pub fn key_strides(
    builder: &Builder<'_>,
    owner: &Interface,
    owner_dims: &[TermId],
) -> Option<Vec<Capacity>> {
    let literal_inner = |dims: &[TermId]| -> Option<Vec<Capacity>> {
        let mut strides = vec![1; dims.len()];
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            let extent = builder.terms.constant_value(dims[axis + 1])?;
            strides[axis] = strides[axis + 1] * extent;
        }
        Some(strides)
    };
    match owner {
        Interface::Store(store) => match &store.lambda.layout {
            crate::phase1::interface::LayoutKind::Box(grid) => {
                let mut strides = vec![1; grid.len()];
                for axis in (0..grid.len().saturating_sub(1)).rev() {
                    strides[axis] = strides[axis + 1] * grid[axis + 1];
                }
                Some(strides)
            }
            _ => literal_inner(owner_dims),
        },
        Interface::Packed { .. } | Interface::Masked { .. } => literal_inner(owner_dims),
        Interface::Ref(reference) => {
            let _ = reference;
            literal_inner(owner_dims)
        }
        _ => None,
    }
}

/// The extent of one axis of a view: the literal of CONST, or the term whose
/// wire the `Ref` node raises (`mapping/README.md` §2.4).
pub fn extent(builder: &Builder<'_>, dimension: TermId) -> Extent {
    match builder.terms.constant_value(dimension) {
        Some(value) => Extent::Literal(value),
        None => Extent::Term(dimension),
    }
}

/// `L`, the normalised lower endpoint of a slice, read back out of the
/// length term phase 0 built (`../../phase0/rules/helpers.rs`, `range_len`):
///
/// ```text
/// len = max(0, (u − L + |s| − 1) div |s|)      s > 0
/// len = max(0, (L − u − |s| − 1) div |s|)      s < 0
/// ```
///
/// `None` when the length term has another shape, and then no rule of this
/// group offers a node for that axis.
pub fn slice_lower(builder: &Builder<'_>, length: TermId, step: i64) -> Option<TermId> {
    slice_bounds(builder, length, step).map(|(lower, _)| lower)
}

/// The normalised endpoints `(L, U)` of a slice, read back out of the length
/// term phase 0 built ([`slice_lower`]).
///
/// `representation/build_masked.md`'s `slice` variant cuts the base's
/// validity at both of them, so it needs the pair; every other consumer of
/// this module needs `L` alone.
pub fn slice_bounds(builder: &Builder<'_>, length: TermId, step: i64) -> Option<(TermId, TermId)> {
    let terms = &builder.terms;
    let magnitude = step.unsigned_abs() as i128;
    let quotient = match terms.node_at(length.index()) {
        NodeView::Max(a, b) => {
            let zero = |at: u32| matches!(terms.node_at(at as usize), NodeView::Const(0));
            if zero(a) {
                b
            } else if zero(b) {
                a
            } else {
                return None;
            }
        }
        _ => length.index() as u32,
    };
    let biased = if magnitude == 1 {
        quotient
    } else {
        match terms.node_at(quotient as usize) {
            NodeView::Div(inner, k) if k == magnitude => inner,
            _ => return None,
        }
    };
    let span = if magnitude == 1 {
        biased
    } else {
        match terms.node_at(biased as usize) {
            NodeView::Add(a, b)
                if matches!(terms.node_at(b as usize), NodeView::Const(n) if n == magnitude - 1) =>
            {
                a
            }
            _ => return None,
        }
    };
    match terms.node_at(span as usize) {
        // `u − L` for a positive step, `L − u` for a negative one.
        NodeView::Sub(upper, lower) => {
            let (lower, upper) = if step > 0 { (lower, upper) } else { (upper, lower) };
            Some((terms.int_at(lower)?, terms.int_at(upper)?))
        }
        _ => None,
    }
}

/// What `MAKE_REF`'s P1 decided for one owner axis of a view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Start {
    /// CONST decided every parameter of the entry: the literal `L` of a
    /// slice, `c⁺` of an index, the extent of a broadcast axis.
    Literal(Capacity),
    /// The parameter is a wire: `MAKE_REF_RUNTIME`'s case.
    Runtime(Runtime),
}

impl Start {
    pub fn literal(&self) -> Option<Capacity> {
        match self {
            Start::Literal(value) => Some(*value),
            Start::Runtime(_) => None,
        }
    }
}

/// The three run-time parameters `make_ref_runtime.md` admits in step 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Runtime {
    /// R1: the normalised start `L` of a slice, whose wire `TERM_TO_WIRES`
    /// answers.
    Start(TermId),
    /// R2: the normalised index `⟨i, d⟩`, whose wire `INDEX_NORMALIZE`
    /// answers.
    Index { index: crate::phase1::graph::Target, extent: TermId },
    /// R5: the broadcast test `¬(dims_u[a] = 1)`.
    Broadcast { extent: TermId },
}

/// P1 of `make_ref.md`: one CONST question per entry, giving each owner axis
/// the parameter of the entry that sits on it.
///
/// `None` when an entry is one no rule of this group covers — a subscript
/// index whose term is not constant is `GATHER`'s.
pub fn parameters(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    matched: &Match,
    epoch: u32,
    under: Condition,
) -> Option<Vec<Start>> {
    match &matched.view {
        // None of the three has a per-axis parameter: the permutation, the
        // C-order position and the diagonal are structural (`make_ref.md`,
        // "transpose", "reshape", "diagonal").
        View::Transpose(_) | View::Reshape | View::Diagonal => Some(Vec::new()),
        View::Broadcast => {
            let mut out = Vec::with_capacity(matched.owner_dims.len());
            for &dimension in &matched.owner_dims {
                let answer =
                    builder.ask_for(candidate, Question::Const(dimension), epoch, under);
                out.push(match answer {
                    Verdict::Constant(extent) => Start::Literal(extent),
                    _ => Start::Runtime(Runtime::Broadcast { extent: dimension }),
                });
            }
            Some(out)
        }
        View::Subscript(entries) => {
            let node = builder.node(&matched.value).cloned();
            let written = node.as_ref().and_then(|node| node.entries().map(|e| e.to_vec()));
            let mut out = Vec::with_capacity(matched.owner_dims.len());
            let (mut owner_axis, mut output_axis, mut position) = (0usize, 0usize, 0usize);
            for entry in entries {
                // The AST entry the normalised entry came from, for the
                // expression index of a written endpoint. A normalised entry
                // the Ellipsis expanded has none.
                let source = written.as_ref().and_then(|written| written.get(position));
                match entry {
                    Entry::NewAxis => {
                        output_axis += 1;
                        position += 1;
                        continue;
                    }
                    Entry::Whole => out.push(Start::Literal(0)),
                    Entry::Slice { step } => {
                        let start = slice_start(
                            builder, candidate, matched, source, owner_axis, output_axis, *step,
                            epoch, under,
                        )?;
                        out.push(start);
                    }
                    Entry::Index => {
                        let start =
                            index_start(builder, candidate, matched, source, owner_axis, epoch, under)?;
                        out.push(start);
                    }
                }
                if !matches!(entry, Entry::Index) {
                    output_axis += 1;
                }
                owner_axis += 1;
                position += 1;
            }
            Some(out)
        }
    }
}

/// `L = norm(t_lo, dims_u[a])` of `make_ref.md` P1, as a literal when CONST
/// decides both the endpoint and the extent, and otherwise as the term the
/// length carries ([`slice_lower`]).
#[allow(clippy::too_many_arguments)]
fn slice_start(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    matched: &Match,
    source: Option<&SubscriptEntry>,
    owner_axis: usize,
    output_axis: usize,
    step: i64,
    epoch: u32,
    under: Condition,
) -> Option<Start> {
    // An omitted endpoint is Python's default and needs no question about
    // the endpoint itself: 0 for a positive step and d − 1 for a negative
    // one (`../../phase0/rules/helpers.rs`, `norm`). The first is a literal
    // whatever the extent is; the second is a term over the extent, which is
    // a literal when CONST decides the extent and `MAKE_REF_RUNTIME`'s R1
    // otherwise — so a reversal of a run-time axis has a node.
    let written = match source {
        Some(SubscriptEntry::Slice { lo: Some(index), .. }) => Some(*index),
        _ => None,
    };
    let extent_term = matched.owner_dims[owner_axis];
    if written.is_none() {
        if step > 0 {
            return Some(Start::Literal(0));
        }
        return match builder.ask_for(candidate, Question::Const(extent_term), epoch, under) {
            Verdict::Constant(value) => Some(Start::Literal(value - 1)),
            _ => {
                let one = builder.terms.constant(1);
                let lower = builder.terms.sub(extent_term, one);
                Some(Start::Runtime(Runtime::Start(lower)))
            }
        };
    }
    // The term of the run-time start, read back out of the length term, for
    // wherever a CONST answer below does not decide the parameter.
    let runtime = slice_lower(builder, matched.dims[output_axis], step)
        .map(|term| Start::Runtime(Runtime::Start(term)));
    let extent = match builder.ask_for(candidate, Question::Const(extent_term), epoch, under) {
        Verdict::Constant(value) => value,
        _ => return runtime,
    };
    let Some(index) = written else {
        return runtime;
    };
    let endpoint = builder
        .sibling(&matched.value, index)
        .and_then(|entry| entry.desc.as_scalar())
        .and_then(|scalar| scalar.term.as_int());
    let Some(endpoint) = endpoint else {
        return runtime;
    };
    match builder.ask_for(candidate, Question::Const(endpoint), epoch, under) {
        Verdict::Constant(value) => Some(Start::Literal(norm(value, extent, step < 0))),
        _ => runtime,
    }
}

/// `c⁺ = ite(t_c < 0, dims_u[a] + t_c, t_c)` of `make_ref.md` P1.
///
/// A term that is not constant is `GATHER`'s and no case of this group, so
/// the whole match is dropped.
fn index_start(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    matched: &Match,
    source: Option<&SubscriptEntry>,
    owner_axis: usize,
    epoch: u32,
    under: Condition,
) -> Option<Start> {
    let index = source.and_then(|entry| entry.expr_index())?;
    let target = builder
        .sibling(&matched.value, index)
        .map(|entry| entry.key.clone())
        .map(crate::phase1::graph::Target::Value)?;
    let term = builder
        .sibling(&matched.value, index)
        .and_then(|entry| entry.desc.as_scalar())
        .and_then(|scalar| scalar.term.as_int())?;
    let Verdict::Constant(value) =
        builder.ask_for(candidate, Question::Const(term), epoch, under)
    else {
        return None;
    };
    if value >= 0 {
        return Some(Start::Literal(value));
    }
    let extent = matched.owner_dims[owner_axis];
    match builder.ask_for(candidate, Question::Const(extent), epoch, under) {
        Verdict::Constant(d) => Some(Start::Literal(d + value)),
        // A negative literal index on a run-time extent: R2, whose wire
        // `INDEX_NORMALIZE` answers.
        _ => Some(Start::Runtime(Runtime::Index { index: target, extent })),
    }
}

/// `norm(b, d)` of `../../phase0/rules/helpers.rs` on literals.
fn norm(b: Capacity, d: Capacity, negative_step: bool) -> Capacity {
    let wrapped = if b < 0 { d + b } else { b };
    let (lower, upper) = if negative_step { (-1, d - 1) } else { (0, d) };
    lower.max(upper.min(wrapped))
}

/// The parameter of one owner axis as the map uses it: a literal, or the
/// index of a wire in `π` (`make_ref.md`, `make_ref_runtime.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapParam {
    Literal(Capacity),
    Wire(usize),
}

impl MapParam {
    /// The map fragment the parameter contributes.
    pub fn expression(self) -> Mu {
        match self {
            MapParam::Literal(value) => Mu::Const(value),
            MapParam::Wire(index) => Mu::Param(index),
        }
    }

}

/// The μ of a subscript over the owner axes (`make_ref.md`,
/// `produced_output_bundle`), given the lower endpoint of each slice entry.
///
/// `lower[a]` is `L` of the entry on owner axis `a` as the caller resolved
/// it: a literal, or the index of a parameter wire in `π`.
pub fn subscript_map(entries: &[Entry], lower: &[Option<Mu>]) -> Vec<Mu> {
    let mut map = Vec::new();
    let (mut owner_axis, mut output_axis) = (0usize, 0usize);
    for entry in entries {
        match entry {
            Entry::Whole => {
                map.push(Mu::Coord(output_axis));
                owner_axis += 1;
                output_axis += 1;
            }
            Entry::Slice { step } => {
                let start = lower[owner_axis].clone().unwrap_or(Mu::Const(0));
                let stride = if *step == 1 {
                    Mu::Coord(output_axis)
                } else {
                    Mu::Scale(*step as Capacity, Box::new(Mu::Coord(output_axis)))
                };
                // `0 + s × i_k` folds to the stride alone, which is what a
                // whole-axis entry is.
                map.push(match &start {
                    Mu::Const(0) => stride,
                    _ => Mu::Add(Box::new(start), Box::new(stride)),
                });
                owner_axis += 1;
                output_axis += 1;
            }
            Entry::Index => {
                map.push(lower[owner_axis].clone().unwrap_or(Mu::Const(0)));
                owner_axis += 1;
            }
            // `None` adds an output axis of extent 1 and no owner axis; its
            // coordinate is 0 and does not occur in μ.
            Entry::NewAxis => output_axis += 1,
        }
    }
    map
}
