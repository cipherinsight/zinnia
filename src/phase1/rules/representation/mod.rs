//! Rule group `representation`
//! (`revision/phase1/rules/representation/README.md`).
//!
//! The rules that decide how the elements of a value are held: as a dense
//! payload built from a batch of reads (`BUILD_DENSE`), in place behind
//! validity bits over another bundle's slots (`BUILD_MASKED`), packed into
//! the first positions of a new payload by key (`PACK_BY_KEY`), several
//! values end to end in one payload (`PACK_SEGMENTS`, which `CONCAT`
//! consumes), as a new version of a payload after an in-place write
//! (`SETITEM`), or as the source's payload with one line of one axis
//! dropped (`DELETE`) or one line inserted (`INSERT`). Every rule produces a
//! bundle of the Store family of `../README.md` §4.1, and `PACK_SEGMENTS`
//! also a `Ref` per segment.
//!
//! **The axis shift `DELETE` and `INSERT` share.** Both match one phase 0
//! transfer rule that changes one axis of one source array by one position
//! and read the same three things off Δ — the source, the axis and the
//! removed or inserted position — so the match, the position and the ports
//! that do not depend on which of the two it is live here, and each rule
//! file states its own map, bundle and emitter.

pub(crate) mod build_dense;
pub(crate) mod build_masked;
pub(crate) mod concat;
pub(crate) mod delete;
pub(crate) mod insert;
pub(crate) mod pack_by_key;
pub(crate) mod pack_segments;
pub(crate) mod setitem;

use crate::phase0::desc::{ArrayDesc, Dtype, ValueKey};
use crate::phase0::rules::Rule as Phase0Rule;
use crate::phase0::source::Producer;
use crate::phase0::term::TermId;

use crate::phase1::build::Builder;
use crate::phase1::cost::{FloatOp, Primitive};
use crate::phase1::graph::Target;
use crate::phase1::interface::{Capacity, Family};
use crate::phase1::oracle::answers::Verdict;
use crate::phase1::oracle::questions::Question;
use crate::phase1::rules::Candidate;
use crate::phase1::table::RowRequest;
use crate::phase1::Reject;

/// What the `source_pattern` of `DELETE` and `INSERT` reads: the source
/// array, the axis whose extent changed, and the value naming the position.
pub(crate) struct AxisShift {
    /// The array the positions are taken out of or put into.
    pub source: ValueKey,
    /// The axis whose extent changed by one.
    pub axis: usize,
    /// `obj`, the value that names the position.
    pub object: ValueKey,
    /// The operands after the position: `values` for `INSERT`, the axis
    /// literal for `DELETE`.
    pub rest: Vec<ValueKey>,
}

/// `source_pattern` of `delete.md` and `insert.md`: the producer is that one
/// transfer rule over `[a, obj, …]`, and the axis is the one on which the
/// output's extent term differs from the source's.
///
/// Like `concat.rs`, the axis is read off Δ and not off the AST node: phase 0
/// normalises an omitted `axis` to 0 on a rank-1 array and rejects it above
/// (`../../phase0/walk/arrays.rs`, `position_axis`), so the shape already
/// says which axis moved and no keyword has to be read back.
pub(crate) fn axis_shift(
    builder: &Builder<'_>,
    key: &ValueKey,
    rule: Phase0Rule,
    array: &ArrayDesc,
) -> Option<AxisShift> {
    let entry = builder.entry(key)?;
    let Producer::Applied { rules, operands, .. } = &entry.producer else {
        return None;
    };
    if rules.as_slice() != [rule] {
        return None;
    }
    let source = operands.first()?.clone();
    let object = operands.get(1)?.clone();
    let dims = builder.array(&source)?.dims.clone();
    if dims.len() != array.rank() {
        return None;
    }
    let mut differing = (0..dims.len()).filter(|&axis| dims[axis] != array.dims[axis]);
    let axis = differing.next()?;
    if differing.next().is_some() {
        return None;
    }
    Some(AxisShift {
        source,
        axis,
        object,
        rest: operands[2..].to_vec(),
    })
}

/// The position of `delete.md` and `insert.md` after normalisation: a
/// literal when CONST decided it, otherwise the wire `INDEX_NORMALIZE`
/// gives.
pub(crate) enum Position {
    Literal(Capacity),
    Wire,
}

impl Position {
    /// The variant name the node carries.
    pub fn variant(&self) -> &'static str {
        match self {
            Position::Literal(_) => "const",
            Position::Wire => "runtime",
        }
    }

    /// The normalised position when it is a literal, which the node records
    /// as its `pos` parameter so that phase 3 reads the pick off the node.
    pub fn literal(&self) -> Option<Capacity> {
        match self {
            Position::Literal(value) => Some(*value),
            Position::Wire => None,
        }
    }
}

/// `P1` of `delete.md` and `insert.md`: the axis moves by exactly one
/// position, and the normalised position `j = ite(t < 0, t + d, t)`.
///
/// `shift` is `−1` for a removal and `+1` for an insertion: the premise is
/// that the output's extent on the axis is the source's plus `shift`, which
/// is what makes the number of positions one (`../../phase0/walk/arrays.rs`,
/// `position_count`, sets the extent to `d + shift · m`). `None` when the
/// premise fails, so a `delete` of several positions is no node of the rule.
pub(crate) fn position(
    builder: &mut Builder<'_>,
    probe: &mut Candidate,
    request: &RowRequest,
    matched: &AxisShift,
    source: &ArrayDesc,
    shift: i128,
) -> Result<Option<Position>, Reject> {
    let under = builder.under(request.condition);
    let extent = source.dims[matched.axis];
    let outer = builder
        .array(request.target.as_value().expect("a value row"))
        .map(|array| array.dims[matched.axis]);
    let Some(outer) = outer else {
        return Ok(None);
    };
    let before = builder.ask_for(probe, Question::Const(extent), request.epoch, under);
    let after = builder.ask_for(probe, Question::Const(outer), request.epoch, under);
    match (before, after) {
        (Verdict::Constant(d), Verdict::Constant(e)) if e == (d + shift).max(0) => {}
        // A run-time extent leaves the number of positions unread, and a
        // constant pair that moved by more than one is a multi-position
        // `delete` or `insert`, which the rule does not cover.
        _ => return Ok(None),
    }
    // The position itself: a scalar whose term CONST may decide.
    let Some(term) = builder.scalar(&matched.object).and_then(|scalar| scalar.term.as_int())
    else {
        return Ok(None);
    };
    let Verdict::Constant(value) =
        builder.ask_for(probe, Question::Const(term), request.epoch, under)
    else {
        return Ok(Some(Position::Wire));
    };
    if value >= 0 {
        return Ok(Some(Position::Literal(value)));
    }
    // A negative literal wraps by the extent, which CONST answered above.
    match builder.ask_for(probe, Question::Const(extent), request.epoch, under) {
        Verdict::Constant(d) => Ok(Some(Position::Literal(d + value))),
        _ => Ok(Some(Position::Wire)),
    }
}

/// The `pos` port: `Represent(⟨t, dims_a[k]⟩, IndexNorm)`, absent when the
/// position is a literal. `false` when the port has no provider.
pub(crate) fn attach_position(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    request: &RowRequest,
    matched: &AxisShift,
    source: &ArrayDesc,
    position: &Position,
) -> Result<bool, Reject> {
    if matches!(position, Position::Literal(_)) {
        return Ok(true);
    }
    let row = builder.index_norm_port(
        request,
        Target::Value(matched.object.clone()),
        source.dims[matched.axis],
        request.condition,
    )?;
    let Some(first) = row.entries.first() else {
        return Ok(false);
    };
    candidate.inputs.push(first.requirement);
    candidate.epoch = candidate.epoch.max(first.epoch);
    Ok(true)
}

/// The `shape_j` ports: one `Represent(dims_v[j], Wire[Idx])` per axis of the
/// output whose extent is not constant. The shifted axis is asked with the
/// others, because its extent is the output's and not the source's.
pub(crate) fn attach_shapes(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    request: &RowRequest,
    dims: &[TermId],
    constant: &[bool],
    axis: usize,
) -> Result<bool, Reject> {
    for (at, &dimension) in dims.iter().enumerate() {
        if at != axis && constant.get(at).copied().unwrap_or(false) {
            continue;
        }
        if builder.terms.constant_value(dimension).is_some() {
            continue;
        }
        let row =
            builder.child(request, Target::Term(dimension), Family::Wire, request.condition)?;
        let Some(first) = row.entries.first() else {
            return Ok(false);
        };
        candidate.inputs.push(first.requirement);
        candidate.epoch = candidate.epoch.max(first.epoch);
    }
    Ok(true)
}

/// The `cond` port: the wire of `g`, which gates the validity of a bundle
/// whose capacity came from a MAX under `g` (`rules/README.md` §5).
pub(crate) fn attach_condition(
    builder: &mut Builder<'_>,
    candidate: &mut Candidate,
    request: &RowRequest,
) -> Result<bool, Reject> {
    let row = builder.child(
        request,
        Target::BoolTerm(request.condition),
        Family::Wire,
        builder.terms.tt(),
    )?;
    let Some(first) = row.entries.first() else {
        return Ok(false);
    };
    candidate.inputs.push(first.requirement);
    candidate.epoch = candidate.epoch.max(first.epoch);
    Ok(true)
}

/// The gate one element select is: `select_f` on a Float element, the field
/// `select` otherwise (`rules/README.md` §5).
pub(crate) fn value_gate(dtype: Dtype) -> Primitive {
    if super::sources::is_float(dtype) {
        Primitive::Float(FloatOp::Select)
    } else {
        Primitive::Select
    }
}

/// The cast an inserted element needs when its dtype is not the array's:
/// `float_cast` for an integer or Boolean wire written into a Float payload
/// (`rules/README.md` §7). `None` when the two agree, and `Some(0)` rows
/// when no cast the library states would do it.
pub(crate) fn cast_rows(from: Dtype, into: Dtype) -> Option<Primitive> {
    if from == into {
        return None;
    }
    match (from, into) {
        (Dtype::Bool | Dtype::Idx | Dtype::I64, Dtype::Float) => {
            Some(Primitive::Float(FloatOp::FromInt))
        }
        // Two integer dtypes share the field encoding, so the wire is the
        // same wire and nothing is emitted.
        _ => None,
    }
}
